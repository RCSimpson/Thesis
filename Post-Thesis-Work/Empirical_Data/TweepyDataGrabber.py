from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from http.client import IncompleteRead
from urllib3.exceptions import IncompleteRead as urllib3_incompleteRead
import json
import pickle

keys = dict(
    ACCESS_KEY = '742476163776290816-ZyoPNkEw3O5hYctwIOovJloRyCjs7K5', #keep the quotes, replace with your consumer key
    ACCESS_SECRET = 'oatZz4srdWPULi5FuNQFRhe5b5dJGKU4eqJfGoYef72ca', #keep the quotes, replace this with your consumer secret key
    CONSUMER_KEY = 'SpFIZF56xsYm1lasz7AN36KMR', #keep the quotes, replace this with your access token
    CONSUMER_SECRET = 'TfiIPzvv8pYG8o9ZPIuPNXYGwEfqZFik705C0BbEFl5495L5Yw', #keep the quotes, replace this with your access token secret
)

access_token = keys['ACCESS_KEY']
access_token_secret = keys['ACCESS_SECRET']
consumer_key = keys['CONSUMER_KEY']
consumer_secret = keys['CONSUMER_SECRET']

# Create tracklist with the words that will be searched for
tracklist = ['#doge']
# Initialize Global variable
tweet_count = 0
# Input number of tweets to be downloaded
n_tweets = 100000
file_name = 'doge_tweet_collection_2.p'
tweets = dict()
text = list()
tweet_id = list()
target_id = list()
time = list()

# Create the class that will handle the tweet stream
class StdOutListener(StreamListener):

    def on_data(self, data):
        global file_name
        global tweet_count
        global n_tweets
        global stream
        try:
            print(tweet_count)
            text.append(json.loads(data)['text'])
            target_id.append(json.loads(data)['user']["screen_name"])
            tweet_id.append(json.loads(data)['id'])
            time.append(json.loads(data)['created_at'])
            tweets['texts'] = text
            tweets['target_id'] = target_id
            tweets['tweet_id'] = tweet_id
            tweets['time'] = time
            tweet_count += 1
            if tweet_count < n_tweets:
                if tweet_count%100 == 0:
                    with open(file_name, 'wb') as fp:
                        pickle.dump(tweets, fp, protocol=pickle.HIGHEST_PROTOCOL)
                return True

            else:
                with open(file_name, 'wb') as fp:
                    pickle.dump(tweets, fp, protocol=pickle.HIGHEST_PROTOCOL)
                stream.disconnect()

        except BaseException as e:
            print("Error on_data: %s, Pausing..." % str(e))
            time.sleep(5)
            return True

        except IncompleteRead as e:
            print("http.client Incomplete Read error: %s" % str(e))
            print("~~~ Restarting stream search in 5 seconds... ~~~")
            time.sleep(5)
            # restart stream - simple as return true just like previous exception?
            return True

        except urllib3_incompleteRead as e:
            print("urllib3 Incomplete Read error: %s" % str(e))
            print("~~~ Restarting stream search in 5 seconds... ~~~")
            time.sleep(5)
            return True

    def on_error(self, status):
        time.sleep(5)
        print(status)
        return True

# Handles Twitter authetification and the connection to Twitter Streaming API
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
print('Authorized')
print(' ')
stream = Stream(auth, l)
print('Stream initialized')
print(' ')
stream.filter(track=tracklist, is_async=True)

