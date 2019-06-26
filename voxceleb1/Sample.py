import numpy
import scipy.io.wavfile
import scipy.signal


DEV_MAP = {"dev": 1, "test": 0}

EXPECTED_RATE = 16000

# Define parameters that produce spectrograms of
# length floor(dt*100) per dt seconds.
WINDOW = "hamming"
NPERSEG = 512
NFFT = 1024
NOVERLAP = 353

class Sample:

    def __init__(self, use, uid, fpath):
        self.dev = DEV_MAP[use]
        self.uid = uid
        self.fpath = fpath
        self.rate = None
        self.data = None
        self.spec = None

    def to_list(self):
        return [self.dev, self.uid, self.spec]

    def load(self):
        self.rate, self.data = scipy.io.wavfile.read(self.fpath)
        assert self.rate == EXPECTED_RATE

    def transform(self):
        f, t, spec = scipy.signal.spectrogram(
            self.data,
            self.rate,
            window=WINDOW,
            nperseg=NPERSEG,
            nfft=NFFT,
            noverlap=NOVERLAP
        )
        real = spec[:NPERSEG//2]
        self.spec = numpy.log10(1+real)
