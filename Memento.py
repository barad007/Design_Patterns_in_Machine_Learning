import torch


class Originator:
    def __init__(self, model):
        self._model = model

    def get_state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def create_memento(self):
        state_dict = self.get_state_dict()
        return Memento(state_dict)

    def restore_from_memento(self, memento):
        state_dict = memento.get_state_dict()
        self.set_state_dict(state_dict)


class Memento:
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def get_state_dict(self):
        return self._state_dict


class Caretaker:
    def __init__(self, originator):
        self._originator = originator
        self._memento = None
        self.filename = "model_state.pth"

    def save_state(self):
        state_dict = self._originator.get_state_dict()
        torch.save(state_dict, self.filename)

    def restore_state(self):
        try:
            state_dict = torch.load(self.filename)
            memento = Memento(state_dict)
            self._originator.restore_from_memento(memento)
        except FileNotFoundError:
            print("The file does not exist.")
