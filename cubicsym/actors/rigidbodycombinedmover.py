import numpy as np

class RigidBodyCombinedMover:

    def __init__(self, cubicboundary):
        self.cubicboundary = cubicboundary
        self.min_perturbation = 0.01
        self.max_attempts_to_set_pertubation = 100

    def get_perturbation(self, current_pos, min_, max_):
        for i in range(self.max_attempts_to_set_pertubation):
            perturbation = np.random.default_rng().normal() * 0.5
            # has to be above a certain minimum value
            if abs(perturbation) >= self.min_perturbation:
                # has to be within min and max
                new_pos = current_pos + perturbation
                if new_pos >= min_ and new_pos <= max_:
                    return perturbation
        return 0

    def apply(self, pose):
        for jump, dof in self.cubicboundary.dof_spec.doforder_str:
            min_, max_ = self.cubicboundary.boundaries[jump][dof]["min"],  self.cubicboundary.boundaries[jump][dof]["max"]
            current_pos = self.cubicboundary.dof_spec.get_jumpdof_str_str(pose, jump, dof)
            perturbation = self.get_perturbation(current_pos, min_, max_)
            self.cubicboundary.dof_spec.perturb_jumpdof_str_str(pose, jump, dof, perturbation)
