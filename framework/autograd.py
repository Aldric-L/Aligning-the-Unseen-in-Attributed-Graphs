import torch

class ReplaceForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, replacement):
        return replacement
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
class StopGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return None

def _mlp_forward_with_jacobian(x: torch.Tensor, mlp: torch.nn.Sequential, training_mode: bool = False, dropout: float = 0.0):
    """Forward pass through MLP computing jacobian analytically with dropout support"""
    batch_size = x.size(0)
    input_dim = x.size(-1)

    # Initialize
    activation = x  # [N, L]
    jacobian = torch.eye(input_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)  # [N, L, L]

    # Store dropout masks by doing a forward pass first
    dropout_masks = []
    temp_activation = x
    for layer in mlp:
        if isinstance(layer, torch.nn.Dropout):
            # Capture the dropout mask
            if training_mode:
                temp_out = layer(temp_activation)
                # Compute mask from input/output ratio safely (preserve device/dtype)
                safe_zero = torch.zeros_like(temp_activation)
                mask = torch.where(temp_activation != 0, temp_out / temp_activation, safe_zero)
                dropout_masks.append(mask)
                temp_activation = temp_out
            else:
                dropout_masks.append(None)
        else:
            temp_activation = layer(temp_activation)

    # Now compute jacobian using the captured masks
    dropout_idx = 0
    for layer in mlp:
        if isinstance(layer, torch.nn.Linear):
            # Linear: y = Wx + b, dy/dx = W
            weight = layer.weight        # [out, in]
            bias = layer.bias if layer.bias is not None else 0.0
            activation = torch.matmul(activation, weight.t()) + bias  # [N, out]
            # jacobian: (W @ previous_jacobian) for each batch
            # weight.unsqueeze(0) -> [1, out, in] broadcast to [N, out, in]
            jacobian = torch.matmul(weight.unsqueeze(0), jacobian)  # [N, out, L]

        elif isinstance(layer, torch.nn.ReLU):
            mask = (activation > 0).to(activation.dtype)  # [N, out]
            jacobian = jacobian * mask.unsqueeze(-1)  # [N, out, L]
            activation = torch.relu(activation)

        elif isinstance(layer, torch.nn.LeakyReLU):
            slope = layer.negative_slope
            # create same-dtype tensor for slope
            slope_tensor = torch.tensor(slope, dtype=activation.dtype, device=activation.device)
            mask = torch.where(activation > 0, torch.ones_like(activation), slope_tensor)
            jacobian = jacobian * mask.unsqueeze(-1)
            activation = torch.where(activation > 0, activation, slope_tensor * activation)

        elif isinstance(layer, torch.nn.Tanh):
            tanh_out = torch.tanh(activation)
            derivative = 1 - tanh_out ** 2  # [N, out]
            jacobian = jacobian * derivative.unsqueeze(-1)
            activation = tanh_out

        elif isinstance(layer, torch.nn.Sigmoid):
            sig_out = torch.sigmoid(activation)
            derivative = sig_out * (1 - sig_out)  # [N, out]
            jacobian = jacobian * derivative.unsqueeze(-1)
            activation = sig_out

        elif isinstance(layer, torch.nn.ELU):
            pre_act = activation
            alpha = layer.alpha
            mask = pre_act > 0
            derivative = torch.where(mask, torch.ones_like(pre_act), alpha * torch.exp(pre_act))  # [N, out]
            jacobian = jacobian * derivative.unsqueeze(-1)
            # apply ELU module to get post-activation
            activation = layer(pre_act)

        elif isinstance(layer, torch.nn.Dropout):
            # Use pre-computed dropout mask (only if we have dropout)
            if dropout > 0:
                dropout_mask = dropout_masks[dropout_idx]
                if dropout_mask is not None:
                    # Apply the same mask to activation and jacobian
                    activation = activation * dropout_mask
                    jacobian = jacobian * dropout_mask.unsqueeze(-1)
                dropout_idx += 1

        else:
            # Unsupported layer type
            raise NotImplementedError(f"Layer type {type(layer)} not supported in analytical jacobian")

    # Return activation and jacobian in shapes:
    # activation: [N, H], jacobian: [N, H, L]
    return activation, jacobian
