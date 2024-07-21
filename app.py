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
        self.train_button.background_color = 'blue'
        self.train_button.tint_color = 'white'
        self.train_button.corner_radius = 5
        self.train_button.flex = 'W'
        self.train_button.action = self.train_model
        self.add_subview(self.train_button)

        self.status_label = ui.Label(frame=(padding, self.train_button.y + element_height + padding, element_width, element_height))
        self.status_label.text = 'Status: Idle'
        self.status_label.text_color = 'white'
        self.status_label.flex = 'W'
        self.add_subview(self.status_label)

    def train_model(self, sender):
        url = self.url_field.text
        epochs = int(self.epochs_field.text)
        epoch_step = int(self.epoch_step_field.text)

        self.status_label.text = 'Status: Fetching data...'
        data = fetch_data(url)
        self.status_label.text = 'Status: Tokenizing data...'
        tokens, word2index, index2word = tokenize(data)
        self.status_label.text = 'Status: Training model...'

        def train_and_update():
            nn, word2index, index2word = train_model(url, epochs, epoch_step, 0.01)
            self.status_label.text = 'Status: Training complete.'
            save_model(word2index, index2word, nn)

        threading.Thread(target=train_and_update).start()

class GenerateTextView(ui.View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Generate Text'
        self.background_color = 'black'
        self.flex = 'WH'
        self.setup_ui()

    def setup_ui(self):
        padding = 10
        element_height = 40
        element_width = self.width - 2 * padding

        self.seed_field = ui.TextField(frame=(padding, padding, element_width, element_height))
        self.seed_field.placeholder = 'Enter seed text'
        self.seed_field.background_color = 'white'
        self.seed_field.text_color = 'black'
        self.seed_field.border_color = 'gray'
        self.seed_field.corner_radius = 5
        self.seed_field.flex = 'W'
        self.add_subview(self.seed_field)
        
        self.length_field = ui.TextField(frame=(padding, self.seed_field.y + element_height + padding, element_width, element_height))
        self.length_field.placeholder = 'Enter length (e.g., 100 words)'
        self.length_field.background_color = 'white'
        self.length_field.text_color = 'black'
        self.length_field.border_color = 'gray'
        self.length_field.corner_radius = 5
        self.length_field.flex = 'W'
        self.add_subview(self.length_field)

        self.generate_button = ui.Button(frame=(padding, self.length_field.y + element_height + padding, element_width, element_height))
        self.generate_button.title = 'Generate Text'
        self.generate_button.background_color = 'green'
        self.generate_button.tint_color = 'white'
        self.generate_button.corner_radius = 5
        self.generate_button.flex = 'W'
        self.generate_button.action = self.generate_text
        self.add_subview(self.generate_button)

        self.output_text = ui.TextView(frame=(padding, self.generate_button.y + element_height + padding, element_width, self.height - self.generate_button.y - 2 * element_height - padding))
        self.output_text.background_color = 'white'
        self.output_text.text_color = 'black'
        self.output_text.border_color = 'gray'
        self.output_text.corner_radius = 5
        self.output_text.flex = 'WH'
        self.add_subview(self.output_text)

    def generate_text(self, sender):
        seed_text = self.seed_field.text
        length = int(self.length_field.text)
        word2index, index2word, nn = load_model()
        generated_text = generate_text(seed_text, length, word2index, index2word, nn)
        self.output_text.text = generated_text

class MainView(ui.View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'PyLLM'
        self.background_color = 'black'
        self.flex = 'WH'
        self.setup_ui()

    def setup_ui(self):
        padding = 10
        element_height = 40
        element_width = self.width - 2 * padding

        self.train_view_button = ui.Button(frame=(padding, padding, element_width, element_height))
        self.train_view_button.title = 'Train Model'
        self.train_view_button.background_color = 'blue'
        self.train_view_button.tint_color = 'white'
        self.train_view_button.corner_radius = 5
        self.train_view_button.flex = 'W'
        self.train_view_button.action = self.show_train_view
        self.add_subview(self.train_view_button)

        self.generate_text_view_button = ui.Button(frame=(padding, self.train_view_button.y + element_height + padding, element_width, element_height))
        self.generate_text_view_button.title = 'Generate Text'
        self.generate_text_view_button.background_color = 'green'
        self.generate_text_view_button.tint_color = 'white'
        self.generate_text_view_button.corner_radius = 5
        self.generate_text_view_button.flex = 'W'
        self.generate_text_view_button.action = self.show_generate_text_view
        self.add_subview(self.generate_text_view_button)

    def show_train_view(self, sender):
        train_view = TrainView()
        train_view.present('fullscreen')

    def show_generate_text_view(self, sender):
        generate_text_view = GenerateTextView()
        generate_text_view.present('fullscreen')

if __name__ == '__main__':
    main_view = MainView()
    main_view.present('fullscreen')