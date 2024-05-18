class Message:
    def __init__(self):
        self._messages = []
        
    def add_message(self, question, answer=None):
        quension_msg_dict = {'from': 'human'}
        quension_msg_dict['value'] = question
        answer_msg_dict = {'from': 'gpt'}
        answer_msg_dict['value'] = answer
        self._messages.append(quension_msg_dict)
        self._messages.append(answer_msg_dict)
    
    @property
    def messages(self):
        return self._messages