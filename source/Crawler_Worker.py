import requests
from bs4 import BeautifulSoup
import ast
import re
import pandas as pd
import numpy as np
import eventlet
import random
import time
from random import randrange
from proxy_bonanza.client import ProxyBonanzaClient
import os
from datetime import datetime
from celery import Celery, group, chord, signature
import json
import redis


columns = ['rank','asin','stars','reviews','answers','positive votes','negative votes','price','percent discount','number sellers']

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:53.0) Gecko/20100101 Firefox/53.0',
    'Mozilela/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0',
]


def _get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
    }

client = ProxyBonanzaClient("XsCGvhtdQV8jWT4C6E2XmRt2qzSt64RALVXKLM2QqxwxzkfJM4!43734")
proxies = [proxy for package in client.get_user_packages() for proxy in client.get_proxies(package['id']) ]


app = Celery('Crawler_Worker', broker="amqp://guest:guest@localhost//", backend="redis://localhost:6379/0")

r = redis.StrictRedis(host='localhost', port=6379, db=1)


def _get_proxy():
    proxy = random.choice(proxies)
    proxy_str = '//{}:{}@{}:{}/'.format(
        proxy['login'], proxy['password'], proxy['ip'],
        proxy['port_http'],
    )
    # print("Using proxy with IP = {}".format(proxy['ip']))

    return {
        'http': 'http:{}'.format(proxy_str),
        'https': 'https:{}'.format(proxy_str),
    }


def get_html_text(url):
    try:
        response = requests.get(url,headers=_get_headers(),proxies=_get_proxy(),timeout=7)
        return response.text
    except:
        return None


def get_tree_from_url(url):
    tree = None
    count = 0

    while tree is None and count < 5:
        html = get_html_text(url)
        if html is None:
            count += 1
            continue

        tree = BeautifulSoup(html,'lxml')

        if tree is None or len(tree.find_all("form" ,{"action":"/errors/validateCaptcha"})) > 0:
            tree = None
        else:
            break

        count = count + 1
    return tree


def count_vote_a_page(tree):
    vote_tags = tree.find_all(class_='vote voteAjax')
    pos_vote = 0
    neg_vote = 0
    for vote_tag in vote_tags:
        a = vote_tag.find(class_='label')
        if a is None:
            return 0,0
        vote_value = int(a.text.split()[0])
        if vote_value < 0:
            neg_vote += vote_value
        else: 
            pos_vote += vote_value
    return pos_vote,neg_vote*-1

def count_votes(asin):
    pos_votes = 0
    neg_votes = 0
    
    vote_url = 'https://www.amazon.com/ask/questions/asin/{0}/{1}/ref=ask_ql_psf_ql_hza'
    
    #get votes from 3 first pages
    for i in range(3):
        vote_link = vote_url.format(asin,i+1)
        print vote_link
        
        tree = get_tree_from_url(vote_link)
        pos_vote,neg_vote = count_vote_a_page(tree)

        pos_votes += pos_vote
        neg_votes += neg_vote
        if tree.find(class_='a-pagination').find("li",{"class":"a-selected"}).find_next_sibling().text.startswith("Next"):
            break

    return pos_votes,neg_votes


@app.task
def get_asins_from_categories(based_url, root_dir_data):
    if not os.path.exists(root_dir_data):
        os.makedirs(root_dir_data)

    print "******************Beauty & Personal Care********************"
    tree = get_tree_from_url(based_url)
    
    tags = tree.find_all(id='zg_browseRoot')[0].find_all('li')[2:]
    tag_urls = [tag.find('a')['href'] for tag in tags]

    for t in tag_urls:
        get_info_from_tag_link.delay(t, root_dir_data)
    print '*****************SETUP TASK DONE***************'

"""
    Get information of products from a tag which link to orther categorys
    Extract urls from tags to get information of product from those urls.
    Input : a tag
            folder_name : directory of file saved information of products
    Output : url extracted from tag, information of products from url extracted
"""
@app.task
def get_info_from_tag_link(tag, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    cate_url = tag
    file_name = cate_url.split('/')[3][20:]
    
    print "******************{0}********************".format(file_name)
    get_cate_asins.delay(cate_url, folder_name, file_name)


@app.task
def finalize_for_category_apage(products, page_no, folder_name, file_name):
    print "Test {}".format(products)
    print "Page number: {} of cate {}".format(page_no, file_name)

    cate_products = r.get(file_name)
    if not cate_products:
        cate_products = []
    else:
        cate_products = json.loads(cate_products)

    cate_products += products

    r.set(file_name, json.dumps(cate_products))

    if len(cate_products) >= 95:
        print "Category -------{}------- is done".format(file_name)
        df_ = pd.DataFrame(cate_products,columns=columns)
        df_.to_csv('{}/{}.csv'.format(folder_name, file_name), sep=',', encoding='utf-8', index_label=None)

"""
    Get asins(100 asins) in one category(which contains 100 products)
    Input : ctg_url stands for url of a category
    Output : list information of products in that category
"""
@app.task
def get_cate_asins(cate_url, folder_name, file_name):
    tree = get_tree_from_url(cate_url)
    page_tags = tree.find_all("a", {"page" : re.compile('[0-9]')})
    cate_asins = []

    r.set(file_name, '[]')
    print "Start with tag: {}".format(file_name)

    group(
        [get_asin_from_apage.s(pt.get('href', ''), pt.get('page'), folder_name, file_name) for pt in page_tags]
    )()

"""
    Get asin and rank of products from a page.
    One page contains 20 products. Best products of 1 category (100 product best saller) need 5 page to show
    Input: url of a page containing 20 products
    Output: dict has form like this {asin:rank} which asin is similar to id, rank is the rank (from 1-100) of product in its category
"""
@app.task
def get_asin_from_apage(page_url, page_no, folder_name, file_name):
    tree = get_tree_from_url(page_url)
        
    data_asin = tree.find_all(class_ = "zg_itemImmersion")
    asins = {}

    for da in data_asin:
        rank = int(da.find(class_='zg_rankNumber').text.replace('.',''))
        asin = ast.literal_eval(da.find(class_='a-section a-spacing-none p13n-asin')['data-p13n-asin-metadata'])['asin']
        asins[asin] = rank

    chord(
        [ get_infor.s(asin, asins[asin]) for asin in asins.keys() ]
    )(finalize_for_category_apage.s(page_no, folder_name, file_name))

"""
    Get information of a product based on its asin
"""
@app.task
def get_infor(asin,rank):
    pdt_tree = get_tree_from_url(product_link + asin)
    print product_link + asin

    if pdt_tree is None:
        return get_infor(asin,rank)
    
    s = pdt_tree.find(id = 'averageCustomerReviews')
    r = pdt_tree.find(id = 'acrCustomerReviewText')
    a = pdt_tree.find(id = "askATFLink")

    if s is None and a is None and r is None:   
        return get_infor(asin,rank)

    temp = '0' if s is None else s.text.split(' ')[0].replace('\n','').strip()
    stars = 0 if re.match("\d+?\.\d+?", temp) is None else float(temp)

    num_reviews = 0 if r is None else int(r.text.split(' ')[0].replace(',',''))
    num_answers = 0 if a is None else int(a.text.replace('\n',' ').strip().split(' ')[0].replace('+',''))

    (pos_votes,neg_votes) = (0,0) if num_answers == 0 else count_votes(asin)

    discount = pdt_tree.find(id = "regularprice_savings")
    percent_discount = 0 if discount is None else int(re.search(r'\((.*?)\)',discount.text).group(1).replace("%",""))

    price_tag = pdt_tree.find(id="priceblock_ourprice")
    price = 0 if price_tag is None else float(price_tag.text.replace("$",""))

    if price == 0:
        price_tag = pdt_tree.find(id='priceblock_saleprice')
        price = 0 if price_tag is None else float(price_tag.text.replace('$',''))

    # updated price 
    if pdt_tree.find(id="snsPrice") is not None:
        price_tag = pdt_tree.find(id="snsPrice").find(class_='a-size-large a-color-price')
        price = 0 if price_tag is None else float(price_tag.text.strip().replace("$",""))

        percent_discount_tag = pdt_tree.find(id="snsPrice").find(class_='a-size-small snsSavings')
        percent_discount = 0 if percent_discount_tag is None else int(re.search(r'\((.*?)\)',percent_discount_tag.text.split()[2]).group(1).replace("%",""))

    sellers_tag = pdt_tree.find(class_='a-size-small aok-float-right')
    num_sellers = 1 if sellers_tag is None else int(sellers_tag.text.split()[0])

    print [rank,asin,stars,num_reviews,num_answers,pos_votes,neg_votes,price,percent_discount,num_sellers]
    return [rank,asin,stars,num_reviews,num_answers,pos_votes,neg_votes,price,percent_discount,num_sellers]


based_url = 'https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty/ref=zg_bs_unv_bt_1_11060711_2'
product_link = 'https://www.amazon.com/dp/'
root_dir_data = '../DATA/data'

if __name__ == '__main__':
    get_asins_from_categories.delay(based_url, root_dir_data + datetime.now().strftime("%Y_%m_%d %H_%M_%S"))