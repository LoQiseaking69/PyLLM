import ui
import threading
from backend import fetch_data, tokenize, train_model, generate_text, save_model, load_model

class TrainView(ui.View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Train Model'
        self.background_color = 'black'
        self.flex = 'WH'
        self.setup_ui()
    
    def setup_ui(self):
        padding = 10
        element_height = 40
        element_width = self.width - 2 * padding

        self.url_field = ui.TextField(frame=(padding, padding, element_width, element_height))
        self.url_field.placeholder = 'Enter URL'
        self.url_field.background_color = 'white'
        self.url_field.text_color = 'black'
        self.url_field.border_color = 'gray'
        self.url_field.corner_radius = 5
        self.url_field.flex = 'W'
        self.add_subview(self.url_field)
        
        self.epochs_field = ui.TextField(frame=(padding, self.url_field.y + element_height + padding, element_width, element_height))
        self.epochs_field.placeholder = 'Total epochs (e.g., 1000)'
        self.epochs_field.background_color = 'white'
        self.epochs_field.text_color = 'black'
        self.epochs_field.border_color = 'gray'
        self.epochs_field.corner_radius = 5
        self.epochs_field.flex = 'W'
        self.add_subview(self.epochs_field)

        self.epoch_step_field = ui.TextField(frame=(padding, self.epochs_field.y + element_height + padding, element_width, element_height))
        self.epoch_step_field.placeholder = 'Epoch step (e.g., 100)'
        self.epoch_step_field.background_color = 'white'
        self.epoch_step_field.text_color = 'black'
        self.epoch_step_field.border_color = 'gray'
        self.epoch_step_field.corner_radius = 5
        self.epoch_step_field.flex = 'W'
        self.add_subview(self.epoch_step_field)

        self.train_button = ui.Button(frame=(padding, self.epoch_step_field.y + element_height + padding, element_width, element_height))
        self.train_button.title = 'Train Model'
        self.train_button.background_color = '#34A853'
        self.train_button.tint_color = 'white'
        self.train_button.corner_radius = 5
        self.train_button.flex = 'W'
        self.train_button.action = self.train_model
        self.add_subview(self.train_button)
        
        self.result_label = ui.Label(frame=(padding, self.train_button.y + element_height + padding, element_width, 80))
        self.result_label.text = 'Training status will appear here'
        self.result_label.text_color = 'white'
        self.result_label.alignment = ui.ALIGN_CENTER
        self.result_label.number_of_lines = 0
        self.result_label.flex = 'W'
        self.add_subview(self.result_label)
        
    def train_model(self, sender):
        url = self.url_field.text
        total_epochs = int(self.epochs_field.text) if self.epochs_field.text.isdigit() else 1000
        epoch_step = int(self.epoch_step_field.text) if self.epoch_step_field.text.isdigit() else 100
        if url:
            threading.Thread(target=self.train_model_thread, args=(url, total_epochs, epoch_step)).start()
            self.result_label.text = 'Training started...'
        else:
            self.result_label.text = 'Please enter a valid URL'
    
    def train_model_thread(self, url, total_epochs, epoch_step):
        self.word2index, self.index2word = train_model(url, total_epochs, epoch_step)
        save_model(self.word2index, self.index2word)
        self.result_label.text = 'Training completed.'
    
    def load_model(self):
        self.word2index, self.index2word, self.nn = load_model()
        if self.word2index is not None:
            self.result_label.text = 'Model loaded successfully.'
        else:
            self.result_label.text = 'No saved model found.'

class GenerateView(ui.View):
    def __init__(self, train_view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Generate Text'
        self.background_color = 'black'
        self.train_view = train_view
        self.flex = 'WH'
        self.setup_ui()
    
    def setup_ui(self):
        padding = 10
        element_height = 40
        element_width = self.width - 2 * padding
        
        self.seed_field = ui.TextField(frame=(padding, padding, element_width, element_height))
        self.seed_field.placeholder = 'Enter seed word'
        self.seed_field.background_color = 'white'
        self.seed_field.text_color = 'black'
        self.seed_field.border_color = 'gray'
        self.seed_field.corner_radius = 5
        self.seed_field.flex = 'W'
        self.add_subview(self.seed_field)
        
        self.generate_button = ui.Button(frame=(padding, self.seed_field.y + element_height + padding, element_width, element_height))
        self.generate_button.title = 'Generate Text'
        self.generate_button.background_color = '#4285F4'
        self.generate_button.tint_color = 'white'
        self.generate_button.corner_radius = 5
        self.generate_button.flex = 'W'
        self.generate_button.action = self.generate_text
        self.add_subview(self.generate_button)
        
        self.result_label = ui.Label(frame=(padding, self.generate_button.y + element_height + padding, element_width, 80))
        self.result_label.text = 'Generated text will appear here'
        self.result_label.text_color = 'white'
        self.result_label.alignment = ui.ALIGN_CENTER
        self.result_label.number_of_lines = 0
        self.result_label.flex = 'W'
        self.add_subview(self.result_label)
        
    def generate_text(self, sender):
        seed_word = self.seed_field.text
        if seed_word and hasattr(self.train_view, 'word2index'):
            generated_text = generate_text(self.train_view.word2index, self.train_view.index2word, self.train_view.nn, seed_word)
            self.result_label.text = f'Generated text: {generated_text}'
        else:
            self.result_label.text = 'Please enter a valid seed word and train the model first'

class NeuralNetworkApp(ui.View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_color = 'black'
        self.train_view = TrainView(frame=self.bounds)
        self.generate_view = GenerateView(self.train_view, frame=self.bounds)
        self.flex = 'WH'
        self.setup_ui()
    
    def setup_ui(self):
        self.seg_control = ui.SegmentedControl(frame=(10, 30, self.width - 20, 32))
        self.seg_control.segments = ['Train Model', 'Generate Text']
        self.seg_control.selected_index = 0
        self.seg_control.action = self.switch_view
        self.seg_control.flex = 'W'
        self.add_subview(self.seg_control)
        
        self.container_view = ui.View(frame=(0, self.seg_control.y + self.seg_control.height + 10, self.width, self.height - (self.seg_control.y + self.seg_control.height + 10)))
        self.container_view.flex = 'WH'
        self.add_subview(self.container_view)
        
        self.views = [self.train_view, self.generate_view]
        self.current_view = None
        self.switch_view()
    
    def switch_view(self, sender=None):
        if self.current_view:
            self.container_view.remove_subview(self.current_view)
        self.current_view = self.views[self.seg_control.selected_index]
        self.current_view.frame = self.container_view.bounds
        self.container_view.add_subview(self.current_view)
        print(f"Switched to view: {self.seg_control.selected_index}")

app = NeuralNetworkApp()
app.present('full_screen')
