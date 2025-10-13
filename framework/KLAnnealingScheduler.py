import numpy as np

class KLAnnealingScheduler:
    """
    Scheduler for KL annealing during training
    """
    def __init__(
        self, 
        kl_weight: float = 1.0,
        anneal_type: str = 'linear',
        anneal_start: float = 0.0,
        anneal_end: float = 1.0,
        anneal_steps: int = 1000,
        name: str = "KLScheduler"
    ):
        """
        Args:
            kl_weight: Maximum KL weight
            anneal_type: Type of annealing ('linear', 'sigmoid', 'cyclical')
            anneal_start: Starting value for annealing
            anneal_end: Final value for annealing
            anneal_steps: Number of steps to reach final value
        """
        self.kl_weight = kl_weight
        self.anneal_type = anneal_type
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
        self.name = name
    
    def step(self):
        """
        Increment step counter
        """
        self.current_step += 1
    
    def get_weight(self) -> float:
        """
        Get current KL weight based on annealing schedule
        
        Returns:
            Current KL weight
        """
        progress = min(1.0, self.current_step / self.anneal_steps)
        
        if self.anneal_type == 'linear':
            weight = self.anneal_start + progress * (self.anneal_end - self.anneal_start)
        
        elif self.anneal_type == 'sigmoid':
            # Sigmoid annealing
            steepness = 5.0  # Controls steepness of sigmoid
            midpoint = 0.5   # Where the sigmoid is centered
            
            # Apply sigmoid function
            sigmoid_progress = 1 / (1 + np.exp(-steepness * (progress - midpoint)))
            weight = self.anneal_start + sigmoid_progress * (self.anneal_end - self.anneal_start)
        
        elif self.anneal_type == 'cyclical':
            # Cyclical annealing (useful for avoiding posterior collapse)
            cycle_length = self.anneal_steps / 4  # 4 cycles in total annealing period
            within_cycle_progress = (self.current_step % cycle_length) / cycle_length
            
            # Ramp up within each cycle, then stay at maximum
            if within_cycle_progress < 0.5:
                cycle_weight = within_cycle_progress * 2
            else:
                cycle_weight = 1.0
                
            weight = self.anneal_start + cycle_weight * (self.anneal_end - self.anneal_start)
        
        else:
            weight = self.kl_weight  # No annealing
        
        return weight * self.kl_weight


class NoKLScheduler(KLAnnealingScheduler):
    """
    Fake Scheduler for KL annealing during training when we do not need KL
    """
    def __init__(self):
        super(NoKLScheduler, self).__init__(kl_weight=0.0,anneal_type= 'linear',anneal_start= 0.0,anneal_end=0.0,anneal_steps=0, name="NoKLScheduler")
    
    def step(self):
        pass
    
    def get_weight(self) -> float:
        return 0.0
