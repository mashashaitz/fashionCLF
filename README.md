This is my interview code.

This is just code, it doesn't work on its own.

For this code to work one would:
  1) add "fashion/___MNIST_DATASET" into the repository.
  2) add a kafka_2.11-1.10 folder
  3) open two terminals and write something like:
    $ cd ../kafka_2.11-1.1.0
    $ kafka_2.11-1.1.0 % bin/zookeeper-server-start.sh config/zookeeper.properties
    
    $ cd ../kafka_2.11-1.1.0
    $ kafka_2.11-1.1.0 % bin/kafka-server-start.sh config/server.properties

    $ cd ../kafka_2.11-1.1.0
    $ kafka_2.11-1.1.0 % bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic fashion
4. The licence to Google Pub/Sub

First I created the two python notebooks to have some fun with the model and learn how to do at least something with Apache Kafka nad Google Pub/Sub. 
Then I made the main file and all other files to combine those.

I don't think I did a great job, but something works.
