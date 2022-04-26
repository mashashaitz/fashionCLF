from google.cloud import pubsub_v1
from google.cloud import pubsub
import os
from concurrent import futures
import numpy as np


class GDealer(object):
    def __init__(self, e_path="fashionclass-1da73e1f76c2.json", project_id="fashionclass", topic_id="fashion"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = e_path

        self.project_id = project_id
        self.topic_id = topic_id

        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub.SubscriberClient()

        self.establish_topic()
        self.establish_reader()

    def establish_topic(self):
        topics = []
        for topic in self.publisher.list_topics(request={"project": "projects/" + self.project_id}):
            topics.append(topic.name)

        topic_path = self.publisher.topic_path(self.project_id, self.topic_id)

        if topic_path not in topics:
            self.publisher.create_topic(request={"name": topic_path})

    def establish_reader(self):
        sub_path = self.subscriber.subscription_path(self.project_id, "fashion_lover")

        current_subs = []
        for subscription in self.subscriber.list_subscriptions(request={"project": "projects/" + self.project_id}):
            current_subs.append(subscription.name)

        if "projects/" + self.project_id + "/subscriptions/fashion_lover" not in current_subs:
            self.subscriber.create_subscription(
                request={"name": sub_path, "topic": "projects/" + self.project_id + "/topics/" + self.topic_id})

    def read(self):
        sub_path = self.subscriber.subscription_path(self.project_id, "fashion_lover")

        response = self.subscriber.pull(
            request={
                "subscription": sub_path,
                "max_messages": 5,
            }
        )

        messages = []
        for msg in response.received_messages:
            messages.append(np.frombuffer(msg.message.data, dtype='float32'))

        ack_ids = [msg.ack_id for msg in response.received_messages]
        if ack_ids:
            self.subscriber.acknowledge(
                request={
                    "subscription": sub_path,
                    "ack_ids": ack_ids,
                }
            )

        return messages

    def write(self, messages):
        path = "projects/" + self.project_id + "/topics/" + self.topic_id

        for message in messages:
            publish_future = self.publisher.publish(path, data=np.ndarray.tobytes(message))
            futures.wait([publish_future], return_when=futures.ALL_COMPLETED)
            publish_future.result(timeout=60)

    def read_write_orders(self, model):
        read_messages = self.read()
        results = []
        for message in read_messages:
            if message.shape == (784,):
                result = model.predict(message.reshape((1, 28, 28, 1)))
                results.append(np.concatenate((message, result), axis=None))

        self.write(results)
