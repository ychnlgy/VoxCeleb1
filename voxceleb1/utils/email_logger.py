from .Logger import Logger

class EmailLogger(Logger):

    def __init__(self, email, log_path):
        super().__init__(log_path)

        import mailupdater
        self.service = mailupdater.Service(email)

        self.num = 0

    def write(self, msg, *args, **kwargs):
        super().write(msg, *args, **kwargs)
        with self.service.create("Update %d" % self.num) as email:
            email.write(msg)
        self.num += 1
