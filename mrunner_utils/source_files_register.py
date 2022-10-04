class SourceFilesRegister:
    def __init__(self):
        self.source_files = []

    def register(self, file):
        self.source_files.append(file)

    def get(self):
        return self.source_files