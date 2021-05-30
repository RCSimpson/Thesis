# So possibly two-ways to go about this.
# 1. Try it the ryan way. Use the streaming api to grab data as it comes in.
# 2. Grab random tweets from Tweepy, and then build it gradually, and build thing out minute to minute
# https://mediaeffectsresearch.wordpress.com/constructing-a-retweet-network/
# Here, we try the second one.

import datetime

today = datetime.date.today()

start_date = today - datetime.timedelta(days = today.weekday(), weeks = 1)
end_date   = start_date + datetime.timedelta(days = 7)

print('Start date:\t%s' % start_date.isoformat())
print('End date:\t%s' % end_date.isoformat())

import time

def utc2snowflake(utc_timestamp):
    return (int(round(utc_timestamp * 1000)) - 1288834974657) << 22

max_id = utc2snowflake(time.mktime(end_date.timetuple()))
min_id = utc2snowflake(time.mktime(start_date.timetuple()))

print('min_id:\t%s' % min_id)
print('max_id:\t%s' % max_id)

import tweepy
import json

CONSUMER_KEY        = 'your_consumer_key'
CONSUMER_KEY_SECRET = 'your_consumer_key_secret'
ACCESS_TOKEN        = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

F_NAME = 'mp_retweet_data_(%s_to_%s).json' % (start_date.isoformat(), end_date.isoformat())

tweets_read = 0
sn_read = 0
with open(F_NAME,'w') as f_out:
    for screen_name in screen_name_list:
        try:
            search = api.user_timeline(screen_name = screen_name, min_id = min_id, max_id = max_id, include_rts = True)
        except tweepy.TweepError:
            pass # can't get records from this user, probably a protected account and it is safe to skip
        else:
            for result in search:
                tweet = result._json
                if tweet.get('retweeted_status'):
                    json.dump(tweet, f_out)
                    f_out.write('\n')
                    tweets_read += 1
            sn_read += 1
            print('\rUsers read:\t%d/%d\tRetweets read:\t%d' % (sn_read, len(screen_name_list), tweets_read))

print('\rUsers read:\t%d/%d\tRetweets read:\t%d\tFinished!' % (sn_read, len(screen_name_list), tweets_read))