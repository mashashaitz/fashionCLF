from pykafka import KafkaClient
import numpy as np


class ADealer(object):
    def __init__(self, host="127.0.0.1:9092", topic="fashion"):
        try:
            self.client = KafkaClient(hosts=host)
        except Exception:
            self.client = KafkaClient(hosts="127.0.0.1:9092")
        try:
            self.topic = self.client.topics[topic]
        except Exception:
            self.topic = self.client.topics["fashion"]

    def read(self):
        consumer = self.topic.get_simple_consumer(consumer_timeout_ms=5000)
        messages = []
        for message in consumer:
            if message is not None:
                messages.append(np.frombuffer(message.value, dtype='float32'))
        return messages

    def write(self, messages):
        with self.topic.get_sync_producer() as producer:
            for message in messages:
                producer.produce(np.ndarray.tobytes(message))

    def read_write_orders(self, model):
        read_messages = self.read()
        results = []
        for message in read_messages:
            if message.shape == (784,):
                result = model.predict(message.reshape((1, 28, 28, 1)))
                results.append(np.concatenate((message, result), axis=None))

        self.write(results)
