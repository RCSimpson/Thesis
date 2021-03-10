import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tweepy
import twitter
import time

keys = dict(
    screen_name = 'LorenzBasin',
    ACCESS_KEY = '742476163776290816-ZyoPNkEw3O5hYctwIOovJloRyCjs7K5', #keep the quotes, replace with your consumer key
    ACCESS_SECRET = 'oatZz4srdWPULi5FuNQFRhe5b5dJGKU4eqJfGoYef72ca', #keep the quotes, replace this with your consumer secret key
    CONSUMER_KEY = 'SpFIZF56xsYm1lasz7AN36KMR', #keep the quotes, replace this with your access token
    CONSUMER_SECRET = 'TfiIPzvv8pYG8o9ZPIuPNXYGwEfqZFik705C0BbEFl5495L5Yw', #keep the quotes, replace this with your access token secret
)


consumer_key = keys['CONSUMER_KEY']
consumer_secret = keys['CONSUMER_SECRET']
access_token = keys['ACCESS_KEY']
access_token_secret = keys['ACCESS_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
screen_name = ""

# elon_tweet_id = "1354618504532684802"
# # space_x_tweet_id = ""
# tesla_tweet_id = "1354702639007739905"
#
# elon = api.get_status(elon_tweet_id)
# tesla = api.get_status(tesla_tweet_id)
# elon_retweeters =api.retweeters(elon_tweet_id)
# tesla_retweeters = api.retweeters(tesla_tweet_id)
#
# disjoint = set(elon_retweeters) - set(tesla_retweeters)
# nodes = tesla_retweeters + list(disjoint)
#
# G = nx.Graph()
# G.add_nodes_from(nodes)
# G.add_node('Elon')
# G.add_node('Tesla')
#
# ee = [(0, i) for i in elon_retweeters ]
# rr = [(1,i) for i in tesla_retweeters]
#
# G.add_edges_from(ee)
# G.add_edges_from(rr)
# pos = nx.spring_layout(G, k=0.2, iterations=20)
# d = dict(G.degree)
# d = np.array([1 + d[node] for node in G.nodes()])
# nx.draw_networkx_edges(G, pos=pos)
# nx.draw_networkx_nodes(G, pos=pos, node_size=200*d/max(d))
# plt.show()

Elon = 0
i=0
elon_ids=[]
tesla_ids = []
dogecoin_ids = []
grimezsz_ids = []

for page in tweepy.Cursor(api.followers_ids, screen_name="elonmusk").pages():
    elon_ids.extend(page)
    print(i)
    i+=1
    if i==2:
        break

i=0
for page in tweepy.Cursor(api.followers_ids, screen_name="Tesla").pages():
    tesla_ids.extend(page)
    print(i)
    i+=1
    if i==1:
        break

i=0
for page in tweepy.Cursor(api.followers_ids, screen_name="dogecoin").pages():
    dogecoin_ids.extend(page)
    print(i)
    i+=1
    if i==1:
        break

i=0
for page in tweepy.Cursor(api.followers_ids, screen_name="Grimezsz").pages():
    grimezsz_ids.extend(page)
    print(i)
    i+=1
    if i==1:
        break

elon_ids = elon_ids[:int(len(elon_ids)/20)]
tesla_ids = tesla_ids[:int(len(tesla_ids)/35)]
dogecoin_ids = tesla_ids[:int(len(dogecoin_ids)/35)]
grimezsz_ids = tesla_ids[:int(len(grimezsz_ids)/35)]

disjoint_set1 = set(elon_ids) - set(tesla_ids) - set(grimezsz_ids) - set(dogecoin_ids)
disjoint_set2 = set(grimezsz_ids) - set(tesla_ids) - set(elon_ids) - set(dogecoin_ids)
disjoint_set3 = set(dogecoin_ids) - set(elon_ids) - set(tesla_ids) - set(grimezsz_ids)
nodes = tesla_ids + list(disjoint_set1) + list(disjoint_set2) + list(disjoint_set3)

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_node('Elon')
G.add_node('Tesla')
G.add_node('Grimes')
G.add_node('Dogecoin')
ee = [('Elon', i) for i in elon_ids ]
rr = [('Tesla',i) for i in tesla_ids]
ss = [('Grimes',i) for i in dogecoin_ids]
tt = [('Dogecoin',i) for i in grimezsz_ids]

labels = {'Elon':'Elon Musk', 'Tesla':'Tesla', 'Dogecoin':'Dogecoin', 'Grimes':'Grimes'}

G.add_edges_from(ee)
G.add_edges_from(rr)
G.add_edges_from(ss)
G.add_edges_from(tt)

d = dict(G.degree)
d = np.array([1 + d[node] for node in G.nodes()])


pos = nx.random_layout(G)
nx.draw_networkx_nodes(G,
                       pos=pos,
                       node_size= 1000*d/max(d),
                       cmap='cool',
                       node_color = d
                       )

nx.draw_networkx_edges(G,
                        pos=pos,
                       arrowsize= 0.01,
                       alpha=0.05
                        )
nx.draw_networkx_labels(G,pos=pos,labels=labels, font_size=10, font_color='k')
plt.show()