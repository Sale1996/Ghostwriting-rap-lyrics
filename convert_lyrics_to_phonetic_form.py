import os
import io
import pickle
import re
from g2p_en import G2p


def load_lyrics_into_map(lyrics_folder_path):
    lyrics = {}
    for filename in os.listdir(lyrics_folder_path):
        isFilenameTxtFileAndNotInLyricsMap = os.path.isfile(lyrics_folder_path + filename) \
                                             and filename.endswith(".txt") \
                                             and not filename in lyrics
        if isFilenameTxtFileAndNotInLyricsMap:
            add_lyrics_to_map(filename, lyrics, lyrics_folder_path)

    return lyrics


def add_lyrics_to_map(filename, lyrics, lyrics_folder_path):
    with open(lyrics_folder_path + filename, "r") as file:
        lyrics[filename] = file.read()


def build_phonetic_form_and_save_lyrics(artist_lyrics_map, author_word_to_phonetic_form_dictionary, author_word_list):
    artist_songs_phonetic_form_map = {}
    for lyrics_filename, lyrics_text in artist_lyrics_map.items():
        phonetic_form_of_lyrics = []
        g2p = G2p()
        lyrics_lines = lyrics_text.split("\n")
        for line in lyrics_lines:
            is_verse_line = "<verse_start>" in line or "<verse_end>" in line
            if is_verse_line:
                phonetic_form_of_lyrics.append(line)
                if "<verse_start>" in line:
                    author_word_list.append("<verse_start>")
                else:
                    author_word_list.append("<verse_end>")
                continue
            phonetic_form_of_line = build_phonetic_form_of_line(g2p, line, author_word_to_phonetic_form_dictionary,
                                                                author_word_list)
            phonetic_form_of_lyrics.append(phonetic_form_of_line)
        artist_songs_phonetic_form_map[lyrics_filename] = make_text_lyrics_of_list_of_lines(phonetic_form_of_lyrics)

    return artist_songs_phonetic_form_map


def build_phonetic_form_of_line(g2p, line, author_word_to_phonetic_form_dictionary, author_word_list):
    phonetic_form_of_line = ""
    words_in_line = line.split(" ")
    for word in words_in_line:
        is_end_of_line_tag = "<end_line>" in word
        if is_end_of_line_tag:
            word, tag = word.split("<")
        phonetic_form_of_word_list = g2p(word)
        phonetic_form_of_word = ""
        for phonetic_char in phonetic_form_of_word_list:
            phonetic_form_of_word += phonetic_char
        phonetic_form_of_line += phonetic_form_of_word
        # save to dictionary
        author_word_to_phonetic_form_dictionary[phonetic_form_of_word] = word.lower()
        author_word_list.append(phonetic_form_of_word)
        phonetic_form_of_line += " "
        if is_end_of_line_tag:
            author_word_list.append("<end_line>")
            phonetic_form_of_line += "<end_line>"

    return phonetic_form_of_line


def make_text_lyrics_of_list_of_lines(artist_song_lyrics_lines):
    filtered_lyrics_of_song = ""
    for line in artist_song_lyrics_lines:
        filtered_lyrics_of_song += line + '\n'
    return filtered_lyrics_of_song


def save_filtered_lyrics_to_destination(artist_lyrics_folder_path, artist_name, lyrics_map):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", artist_name)
    for lyrics_file_name, lyrics_text in lyrics_map.items():
        is_there_any_text_left_in_lyrics = lyrics_text != ""
        if is_there_any_text_left_in_lyrics:
            with open(artist_lyrics_folder_path + artist_name_for_document_name + '/phonetic/' +
                      lyrics_file_name, 'w') as output_stream:
                output_stream.write(lyrics_text)


def save_dictionary(obj, root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_word_list(obj, root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '_word_list.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dictionary(root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_list(root_path, author_name):
    artist_name_for_document_name = re.sub("[^0-9a-zA-Z]+", "_", author_name)
    dictionary_file_path = root_path + artist_name_for_document_name + "/"

    with open(dictionary_file_path + artist_name_for_document_name + '_word_list.pkl', 'rb') as f:
        return pickle.load(f)


'''
MAIN
'''
artist_name = "Jay_Z"
artist_filtered_lyrics_folder_path = "./data/rap_lyrics_links/" + artist_name + '/filtered/'

artist_form_lyrics_folder_path = "./data/rap_lyrics_links/"
author_word_to_phonetic_form_dictionry = {}
author_word_list = []

artist_lyrics_map = load_lyrics_into_map(artist_filtered_lyrics_folder_path)
phonetic_form_map_of_lyrics = build_phonetic_form_and_save_lyrics(artist_lyrics_map,
                                                                  author_word_to_phonetic_form_dictionry,
                                                                  author_word_list)
save_filtered_lyrics_to_destination(artist_form_lyrics_folder_path, artist_name, phonetic_form_map_of_lyrics)

author_word_to_phonetic_form_dictionry["<end_line>"] = "<end_line>"
author_word_to_phonetic_form_dictionry["<verse_start>"] = "<verse_start>"
author_word_to_phonetic_form_dictionry["<verse_end>"] = "<verse_end>"

save_dictionary(author_word_to_phonetic_form_dictionry, artist_form_lyrics_folder_path, artist_name)
save_word_list(author_word_list, artist_form_lyrics_folder_path, artist_name)

loaded_dictionary = load_dictionary(artist_form_lyrics_folder_path, artist_name)
loaded_word_list = load_list(artist_form_lyrics_folder_path, artist_name)
print("complete")
