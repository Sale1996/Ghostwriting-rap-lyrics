# Fabolous
# Eminem
# Jay Z
# Notorius B.I.G
# Lil Wayne
# izabrani na osnovu jednog istrazivanja...
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import re


def parse_and_process_url(url):
    request_result = requests.get(url)
    page_source = request_result.content
    # parsed_page_source_return = BeautifulSoup(page_source, 'lxml')
    parsed_page_source_return = BeautifulSoup(page_source, 'html.parser')
    return parsed_page_source_return


def get_authors_url(list_of_authors, parsed_page_source_input):
    urls = {}
    ohhla_table = parsed_page_source_input.find('table')
    for a_tag in ohhla_table.find_all('a'):
        for artist in list_of_authors:
            give_href_if_author_contains_or_matches_in_tag(a_tag, artist, urls)
    return urls


def give_href_if_author_contains_or_matches_in_tag(tag, artist, urls):
    if fuzz.ratio(artist.lower(), tag.text.lower()) > 95:
        urls[artist] = tag.attrs['href']
    elif artist in tag.text:
        urls[artist] = tag.attrs['href']


def get_author_and_his_song_urls(author_to_url_extension_map):
    author_to_list_of_song_map = {}
    for author_name in author_to_url_extension_map:
        parsed_artist_page_source = parse_and_process_url(ohhla_home_page_url
                                                          + author_to_url_extension_map[author_name])
        document_author_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
        author_songs_url = get_author_song_urls(parsed_artist_page_source)
        with open('./data/rap_lyrics_links/' + document_author_name + '_song_links.txt', 'w') as output_stream:
            output_stream.write(str(author_songs_url))
        author_to_list_of_song_map[author_name] = author_songs_url

    return author_to_list_of_song_map


def get_author_song_urls(artist_page_source):
    table_with_author_albums = artist_page_source.find("table", {"id": "AutoNumber7"})
    author_albums = table_with_author_albums.find_all("p")

    return extract_songs_url_from_albums(author_albums)


def extract_songs_url_from_albums(author_albums):
    author_song_urls = []
    for p_tag in author_albums:
        is_not_correct_p_tag = not p_tag.find("a")
        if is_not_correct_p_tag:
            break
        is_there_not_regular_albums_on = p_tag.find("a").text == "remix"
        if is_there_not_regular_albums_on:
            break
        table_inside_p_tag = p_tag.find("table")
        for a_tag in table_inside_p_tag.find_all("a"):
            if "BUY NOW" in a_tag.text:
                continue
            elif ".txt" in a_tag.attrs['href']:
                author_song_urls.append([a_tag.text, a_tag.attrs['href']])

    return author_song_urls


target_artists = ['Fabolous', 'Eminem', 'Jay-Z', 'Notorious B.I.G', 'Lil Wayne', 'Drake', 'Wu-Tang Clan']

ohhla_home_page_url = "http://ohhla.com/"
ohhla_favorite_url_extension = "favorite.html"
parsed_favorite_page_source = parse_and_process_url(ohhla_home_page_url + ohhla_favorite_url_extension)

ohhla_author_to_url_author_page_extension_map = get_authors_url(target_artists, parsed_favorite_page_source)
ohhla_author_to_list_of_song_url_map = get_author_and_his_song_urls(ohhla_author_to_url_author_page_extension_map)

for author in ohhla_author_to_list_of_song_url_map:
    all_author_songs_url = ohhla_author_to_list_of_song_url_map[author]
    for one_author_song in all_author_songs_url :
        parsed_song_lyrics_page = parse_and_process_url(ohhla_home_page_url + one_author_song[1])
        is_not_correct_pre_tag = not parsed_song_lyrics_page.find("pre")
        if is_not_correct_pre_tag:
            continue
        text_of_song = parsed_song_lyrics_page.find("pre").text

        document_author_name = re.sub("[^0-9a-zA-Z]+", "_", author)
        song_name = re.sub("[^0-9a-zA-Z]+", "_", str(one_author_song[0]))
        with open('./data/rap_lyrics_links/' + document_author_name + '/' +
                  song_name + '.txt', 'w') as output_stream:
            output_stream.write(text_of_song)
