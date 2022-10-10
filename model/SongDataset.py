from torch.utils.data import Dataset


class SongDataset(Dataset):
    def __init__(self, data, sz=None, model_name=None):
        if model_name == 'BaseModel':
            model_name = model_name
        elif model_name != 'HATwithoutFormModel':
            model_name = 'FormModel'
        else:
            model_name = 'TextureModel'

        self.model_name = model_name

        # (#dataset, max_seq, #event)
        self.x = data['x']
        self.y = data['y']
        self.mask = data['mask']

        # Form
        if 'Form' in model_name:
            self.form_index = data['form_index']
            self.form_index_section_len = data['form_index_section_len']
            self.form_index_chord_len = data['form_index_chord_len']

        # Texture
        if 'Texture' in model_name:
            self.texture_index = data['texture_index']
            self.texture_index_chord_len = data['texture_index_chord_len']
            self.texture_index_note_len = data['texture_index_note_len']

        if sz is not None:
            self.x = self.x[:sz]
            self.y = self.y[:sz]
            self.mask = self.mask[:sz]

            if 'Form' in model_name:
                self.form_index = self.form_index[:sz]
                self.form_index_section_len = self.form_index_section_len[:sz]
                self.form_index_chord_len = self.form_index_chord_len[:sz]

            if 'Texture' in model_name:
                self.texture_index = self.texture_index[:sz]
                self.texture_index_chord_len = self.texture_index_chord_len[:sz]
                self.texture_index_note_len = self.texture_index_note_len[:sz]

        print('x: ', self.x.shape)
        print('y: ', self.y.shape)
        print('mask: ', self.mask.shape)

        if 'Form' in model_name:
            print('form_index: ', self.form_index.shape)
            print('form_index_section_len: ',
                  self.form_index_section_len.shape)
            print('form_index_chord_len: ', self.form_index_chord_len.shape)
        if 'Texture' in model_name:
            print('texture_index: ', self.texture_index.shape)
            print('texture_index_chord_len: ',
                  self.texture_index_chord_len.shape)
            print('texture_index_note_len:', self.texture_index_note_len.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        items = [self.x[idx], self.y[idx], self.mask[idx]]
        if 'Form' in self.model_name:
            items.extend(
                [self.form_index[idx], self.form_index_section_len[idx], self.form_index_chord_len[idx]])
        if 'Texture' in self.model_name:
            items.extend([self.texture_index[idx], self.texture_index_chord_len[idx],
                          self.texture_index_note_len[idx]])

        return items
