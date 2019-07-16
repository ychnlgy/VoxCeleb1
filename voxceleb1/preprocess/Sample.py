import numpy
import scipy.io.wavfile
import scipy.signal


EXPECTED_RATE = 16000

# Define parameters that produce spectrograms of
# length floor(dt*100) per dt seconds.
WINDOW = "hamming"
NPERSEG = 512
NFFT = 1024
NOVERLAP = 353

class Sample:

    def __init__(self, dev, uid, fpath, spec=None):
        self.dev = dev
        self.uid = uid
        self.fpath = fpath
        self.spec = spec
        self.rate = EXPECTED_RATE
        self.data = None

    def to_list(self):
        return [self.dev, self.uid, self.fpath, self.spec]

    @staticmethod
    def from_list(obj):
        return Sample(*obj)

    def load(self):
        rate, data = scipy.io.wavfile.read(self.fpath)
        print(self.fpath)
        self.rate, self.data = self._correct_raw_audio(rate, data)
        assert self.rate == EXPECTED_RATE, "File: %s has rate %d" % (
            self.fpath, self.rate
        )

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
        self.spec = numpy.log10(1+real).astype(numpy.float16)
        self.data = None  # release memory

    def _correct_raw_audio(self, rate, data):
        # Handle multi channel, different dtypes
        try:
            iinfo = numpy.iinfo(data.dtype)
        except ValueError:
            iinfo = None

        if iinfo is not None:
            r = iinfo.max - iinfo.min
            data = (
                data.astype(numpy.float32).mean(axis=-1) - iinfo.min
            ) / r * 2 - 1

        if rate != EXPECTED_RATE:

            # Handle different rates
            frac = EXPECTED_RATE/rate
            n = len(data)
            N = int(2**(numpy.floor(numpy.log2(n))+1))
            buf = numpy.zeros(N)
            buf[:n] = data
            M = int(numpy.ceil(frac * N))
            m = int(round(frac * n))
            resampled = scipy.signal.resample(buf, M)
            rate, data = EXPECTED_RATE, resampled[:m]

        print(data.shape)
        return rate, data
