import numpy as np

class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.state = 0.0

    def s_transition(self, x):
        """
        Функция активации на базе S-образного фазового перехода.
        Реализует баланс сил двух витков разного направления.
        """
        # Прямой и обратный витки (зеркальная антисимметрия)
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        
        # S-образный балансирующий переход
        return np.tanh(forward_loop + backward_loop)

    def activate(self, inputs, weights):
        """
        Вычисление состояния нейрона с эффектом самокомпенсации.
        """
        net_input = np.dot(inputs, weights)
        self.state = self.s_transition(net_input)
        return self.state
