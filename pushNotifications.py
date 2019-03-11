from pusher_push_notifications import PushNotifications

beams_client = PushNotifications(
    instance_id='2f23a1d1-dc77-48d8-8474-d7dda1d9ee14',
    secret_key='BD15BDBFF5D0A9D5D9D0A42223D54F72027CAB42C527A2C626B3AE626222EC71',
)

response = beams_client.publish_to_interests(
  interests=['hello'],
  publish_body={
    'apns': {
      'aps': {
        'alert': 'Hello!',
      },
    },
    'fcm': {
      'notification': {
        'title': 'Hello',
        'body': 'Hello, world!',
      },
    },
  },
)

print(response['publishId'])