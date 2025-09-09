#module for functions/classes related to homework-one problem 1
#Lucas Steinberger
#AM 226, 09/08/2025
import numpy as np

class Circuit:
    def __init__(self, state, params, fed = True):
        """Initialize the circuit with a given state
        Args:
            state (array of int): The initial state of the circuit
            params (dict): Dictionary of parameters w, alpha, and B for the circuit"""
        self.state = state
        self.params = params
        self.w = params['w']
        self.B = params['B']
        self.alpha = params['alpha']
        self.fixed = False #has the system found a fixed point yet?
        
        self.neuron_names = ['pLN0', 'pLN1-4', 'uPN', 'mPN', 'CSD']
        # Define the weight matrix for the circuit
        if fed:
            self.CSDW = 1
        elif not fed:
            self.CSDW = 2
        else:
            raise ValueError("fed must be True or False")
        #for ease of code, define local variable _w and _B
        _w = self.w
        _B = self.B
        self.matrix = np.array([
            [0,-1, -_w,-_w,-_w],                #row 1
            [-_w,0,-_w,-1,0],                   #row 2
            [0,0,0,0,1],                        #row 3
            [0,0,0,0,0],                        #row 4
            [-_B, -_w, self.CSDW, -_w, 0]])     #row 5
        
    def step(self):
        """Perform one step of the circuit update"""
        Iorn = np.array([1, 1, 1, self.w, self.w]) #input from ORN
        Ibasal = np.array([0, 0, 0, self.alpha, 0]) #basal input to CSD
        h = np.matmul(self.matrix.transpose(), self.state) + Iorn + Ibasal
        bin_indices = [0,1, 3] #indices of binary neurons
        tri_indices = [2,4]     #indices of trinary neurons
        new_state = np.zeros_like(self.state)
        new_state[bin_indices] = (h[bin_indices] > 0).astype(int)
        new_state[tri_indices] = 0 if h[tri_indices][0] < 0 else (1 if h[tri_indices][0] < 2 else 2)
        
        state_changed = not np.array_equal(self.state, new_state)
        self.state  = new_state
        if not state_changed:
            self.fixed = True
            print("The system has reached a fixed point.")

    def print_state(self):
        """Print the current state of the circuit in a nice table format"""
        print("Current state of the circuit:")
        print("Node | State")
        print("------------")
        for neuron, val in zip(self.neuron_names, self.state):
            print(f"{neuron:4} | {val:5}")
        print(f"y  | {self.y()}")

    def search_fixed_point(self, max_steps=100):
        """Run the circuit until it reaches a fixed point or max_steps is reached
        Args:
            max_steps (int): Maximum number of steps to run the circuit"""
        step_count = 0
        while not self.fixed and step_count < max_steps:
            self.step()
            step_count += 1
        if self.fixed:
            print(f"Fixed point reached in {step_count} steps.")
            self.print_state()
        else:
            print(f"Max steps reached ({max_steps}) without finding a fixed point.")

    def y(self):
        """return the sign of uPN - 3*mPN"""
        return np.sign(self.state[2] - 3*self.state[3])
    
    def inactive_csd(self):
        """Set the CSD neuron to inactive (0)"""
        self.state[4] = 0
        self.fixed = False

    def inactivep_LN14(self):
        """Set the pLN1-4 neuron to inactive (0)"""
        self.state[1] = 0
        self.fixed = False
    def block_pln0_14(self):
        """Set the weights from pLN0 to pLN1-4"""
        self.matrix[1,0] = 0
        self.fixed = False
    def block_pln14_0(self):
        """Set the weight from pLN1-4 to pLN0 to 0"""
        self.matrix[0,1] = 0
        self.fixed = False

    


