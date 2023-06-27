from symmetryhandler.reference_kinematics import perturb_jumpdof_int_int, dof_str_to_int, get_jumpdof_int_int
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof
import numpy as np

class RigidBodyDofAdaptiveMover:
    """Standin for the C++ version of the mover"""

    def __init__(self, name, max_attempts_to_set_pertubation=100):
        self.name = name
        self.jumps = []
        self.dofs = []
        self.param1 = []
        self.param2 = []
        self.min_pertubation = []
        self.max = []
        self.min = []
        self.limit_movements = []
        self.initial_placement = {}
        self.max_attempts_to_set_pertubation = max_attempts_to_set_pertubation

    def add_jump(self, pose, jump_name, dof_name, step_type="gauss", param1=0.5, param2=0.0,
                 min_pertubation=0.01, limit_movement=False, max=None, min=None):
        jumpid = sym_dof_jump_num(pose, jump_name)
        dofid = dof_str_to_int[dof_name]
        assert not self.initial_placement.get(jumpid, {}).get(jumpid, False), "jump name and dof name is already present"
        self.jumps.append(jumpid)
        self.dofs.append(dofid)
        self.param1.append(param1)
        self.param2.append(param2)
        self.min_pertubation.append(min_pertubation)
        self.max.append(max)
        self.min.append(min)
        self.limit_movements.append(limit_movement)
        current_pertubation = get_jumpdof_int_int(pose, jumpid, dofid)
        if jump_name in self.initial_placement:
            self.initial_placement[jumpid][dofid] = current_pertubation
        else:
            self.initial_placement[jumpid] = {dofid: current_pertubation}

    def get_initial_placement(self, pose, jumpid, dofid):
        return get_jumpdof_int_int(pose, jumpid, dofid)

    def generate_perturbation(self, param1, param2, min_perturbation=None):
        for i in range(self.max_attempts_to_set_pertubation):
            val = np.random.default_rng().normal() * param1 + param2
            if min_perturbation is not None:
                if val < min_perturbation:
                    continue
            return val
        return 0

    def apply(self, pose):
        for jumpid, dofid, param1, param2, min_pertubation, limit_movement, max_, min_ in zip(self.jumps, self.dofs,
                                                                                              self.param1, self.param2,
                                                                                              self.min_pertubation,
                                                                                              self.limit_movements, self.max, self.min):
            if limit_movement:
                initial_pertubation = self.get_initial_placement(pose, jumpid, dofid)
                current_pertubation = get_jumpdof_int_int(pose, jumpid, dofid)
                diff = current_pertubation - initial_pertubation
                for i in range(self.max_attempts_to_set_pertubation):
                    pertubation = self.generate_perturbation(param1, param2, min_pertubation)
                    diff_plus_pertubation = diff + pertubation
                    if diff_plus_pertubation <= max_ and diff_plus_pertubation >= min_:
                        perturb_jumpdof_int_int(pose, jumpid, dofid, pertubation)
                        break
            else:
                pertubation = self.generate_perturbation(param1, param2, min_pertubation)
                perturb_jumpdof_int_int(pose, jumpid, dofid, pertubation)
