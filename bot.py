from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core import utils
from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels import HttpInputChannel
from rasa_core.channels.facebook import FacebookInput
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


class RestaurantPolicy(KerasPolicy):
    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model


def train_dialogue(domain_file="data/servicing-bot/domain.yml",
                   model_path="data/servicing-bot/dialogue",
                   training_data_file="data/servicing-bot/story/stories.md"):
    agent = Agent(domain_file,policies=[MemoizationPolicy(), KerasPolicy()])
    train_nlu()
    agent.train(
            training_data_file,
                max_history=3,
                epochs=100,
                batch_size=50,
                augmentation_factor=50,
                validation_split=0.2
            
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data('data/servicing-bot/nlu/rasa-servicing.json')
    trainer = Trainer(RasaNLUConfig("configs/config_servicing.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('data/servicing-bot/', fixed_model_name="current")

    return model_directory


def run(serve_forever=True,port=5002):
    
    interpreter = RasaNLUInterpreter("data/servicing-bot/rasa_servicing_en_nlu/current")
    agent = Agent.load("data/servicing-bot/dialogue", interpreter=interpreter)
    input_channel = FacebookInput(
                                  fb_verify="rasa_bot",  # you need tell facebook this token, to confirm your URL
                                  fb_secret="customerserviceagent100292",  # your app secret
                                  fb_tokens={"1951390505111264": "EAADAwrooCtoBAJ5cUzCya4UG8vLG7TBXrXJoNoeAplZCadQlePYIwqQdgi2EPrlhRQkntyfmQcUrOYZAgBZByPZAqJQvLRc0N4wjNCSJUx7D4D8OvuRctMP3HcHughaM4l2ZCtmvaZCCr040MK1aTwkx2wmQ6EfQwKo2qrJkMZAhwZDZD"},   # page ids + tokens you subscribed to
                                  debug_mode=True    # enable debug mode for underlying fb library
                                  )
    if serve_forever:
        agent.handle_channel(HttpInputChannel(port, "/app", input_channel))
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide the port')
    parser.add_argument('--port', type=str, help='Port number', required=True)
    port = parser.parse_args().port
    run(port)
    exit(1)
