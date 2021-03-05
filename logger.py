class logger():
    def __init__(self, filedir):
        self.filedir = filedir
        self.buffers = []

    def flush(self):
        with open(self.filedir, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buffers))
            f.write('\n')
        self.buffers = []


    def write(self,msg):
        print(msg)
        self.buffers.append(msg)
        self.flush()
