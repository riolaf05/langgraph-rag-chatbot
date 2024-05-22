import os
import boto3
import logging
from typing import Optional

from config.environments import (
    SNS_TOPIC,
    SNS_ENDPOINT_SUBSCRIBE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
)


class SubscriptionManager(object):
    """
    Class to create and delete SNS subscriptions for an endpoint.
    """

    _subscription_arn: Optional[str]

    def __init__(self, endpoint: str):
        self._subscription_arn = None
        self._endpoint = endpoint

    @property
    def subscription_arn(self) -> str:
        return self._subscription_arn

    @subscription_arn.setter
    def subscription_arn(self, arn: str):
        self._subscription_arn = arn

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def create_subscription(self):
        """
        Subscribes an endpoint of this app to an SNS topic.

        :return: None
        """
        logging.info("Subscribing to SNS")
        sns = boto3.client(
            "sns",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

        response = sns.subscribe(
            TopicArn=SNS_TOPIC,
            Protocol="http",
            # TODO FORCE application/json content type, default is plain/text and it does not works!!
            Endpoint=f"{SNS_ENDPOINT_SUBSCRIBE}/{self.endpoint}",
        )
        print("SUBSCRIBE RESPONSE\n", response)
        self._subscription_arn = response["SubscriptionArn"]

    def delete_subscription(self):
        """
        Deletes the endpoint subscription.

        :return: None
        """
        logging.info("Unsubscribing from SNS")
        sns = boto3.client(
            "sns",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        sns.unsubscribe(self.subscription_arn)

    def confirm_subscription(self, token: str):
        """
        Confirms the subscription to the topic, given the confirmation token.

        :param token:
        :return:
        """
        logging.info("Confirming Subscription")
        sns = boto3.client(
            "sns",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        response = sns.confirm_subscription(TopicArn=SNS_TOPIC, Token=token)
        self._subscription_arn = response["SubscriptionArn"]
