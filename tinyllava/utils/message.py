'''
@Description: 
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
class Message:
    def __init__(self, msg=None):
        self._messages = msg if msg else []
        self._images = []
        self.skip_next = False
        
    def add_message(self, question, answer=None):
        quension_msg_dict = {'from': 'human'}
        quension_msg_dict['value'] = question
        answer_msg_dict = {'from': 'gpt'}
        answer_msg_dict['value'] = answer
        self._messages.append(quension_msg_dict)
        self._messages.append(answer_msg_dict)
        
    def add_image(self, image, index=0):
        self._images.append((image, index))
        
    @property
    def images(self):
        return self._images    
    
    @property
    def messages(self):
        return self._messages
    
    def copy(self):
        return Message(self._messages)
    
    def to_gradio_chatbot(self):
        ret = []
        for i, msg in enumerate(self.messages):
            if i % 2 == 0:
                if len(self.images) != 0 and i == self.images[0][1]:
                    image = self.images[0][0]
                    import base64
                    from io import BytesIO
                    msg = msg['value']
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg['value'], None])
            else:
                ret[-1][-1] = msg['value']
        return ret