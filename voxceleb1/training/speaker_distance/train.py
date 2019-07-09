from .data_producer import SubjectDataProducer

def train(config, producer, model, log):
    
    data_producer = SubjectDataProducer(
        slice_size=
