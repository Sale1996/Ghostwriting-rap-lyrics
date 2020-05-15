import os
import io
import re


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


def filter_artist_verses_and_return_map(artist_lyrics_map, artist_name):
    filtered_artist_lyrics_map = {}
    for lyrics_filename, lyrics_text in artist_lyrics_map.items():
        print(lyrics_filename)
        print("=" * 80)
        lyrics_text_as_buffer = io.StringIO(lyrics_text)
        lyrics_lines = lyrics_text_as_buffer.readlines()

        lyrics_lines_without_desc = remove_lyrics_description_lines(lyrics_lines)
        verses = get_list_of_verses_in_lyrics(lyrics_lines_without_desc)
        artist_verses = filter_artist_verses(artist_name, verses)
        artist_verses_with_tags = add_learning_tags_to_verse(artist_verses)
        artist_cleaned_verses = clear_artist_verses_of_background_voices_and_short_lines(artist_verses_with_tags)
        filtered_lyrics_of_song = make_text_lyrics_of_list_of_verses(artist_cleaned_verses)
        print(filtered_lyrics_of_song)

        filtered_artist_lyrics_map[lyrics_filename] = filtered_lyrics_of_song

    return filtered_artist_lyrics_map


def remove_lyrics_description_lines(lyrics_lines):
    if "*" in lyrics_lines[4][-5:]:
        number_of_desc_lines = 7
    else:
        number_of_desc_lines = 5
    lyrics_lines_without_desc = lyrics_lines[number_of_desc_lines:]
    return lyrics_lines_without_desc


def get_list_of_verses_in_lyrics(lyrics_lines_without_desc):
    verse = []
    verses = []
    for line in lyrics_lines_without_desc:
        line = remove_multiple_endline(line)
        blank_between_verses = ""
        if line == blank_between_verses:
            if len(verse) > 4:
                verses.append(verse)
            verse = []
        else:
            verse.append(line)
    return verses


def remove_multiple_endline(line):
    if line.endswith("\n"):
        line = line.replace("\n", "")
    return line


def filter_artist_verses(artist_name, verses):
    artist_verses = []
    for verse in verses:
        first_line_of_verse_lowercase = verse[0].lower()
        if is_artist_verse(first_line_of_verse_lowercase, artist_name):
            artist_verses.append(verse)
        else:
            continue

    return artist_verses


def is_number_of_white_spaces_less_than_two(first_line_of_verse_lowercase):
    '''
        Number of white spaces represent special case when we have:
            -"Verse 1", "Verse 2", ...  [one space]
            -"Verse one", "Verse two",  [one space]
            -"Verse 1 Artist_name" , "Verse one Artist_name" [two spaces, but our Artist]
            -"Verse 1 Other_artist_name" ... etc [two spaces, other Artist!]
        Only verse with last case should not be parsed and passed to parsed_song list
    '''
    isArtistVerseTag = 'verse' in first_line_of_verse_lowercase and first_line_of_verse_lowercase.count(' ') < 2
    return isArtistVerseTag


def is_artist_verse(first_line_of_verse_lowercase, artist_name):
    is_first_line_header_of_verse = is_verse_header(first_line_of_verse_lowercase)
    if is_first_line_header_of_verse:
        is_our_artist_name_in_verse_head = artist_name.lower() in first_line_of_verse_lowercase
        is_intro_or_outro = "intro" in first_line_of_verse_lowercase or "outro" in first_line_of_verse_lowercase
        is_hook = "hook" in first_line_of_verse_lowercase
        is_chorus = "chorus" in first_line_of_verse_lowercase
        isNumberOfWhiteSpacesLessThanTwo = is_number_of_white_spaces_less_than_two(first_line_of_verse_lowercase)

        if is_intro_or_outro or is_hook or is_chorus:
            return False
        elif is_our_artist_name_in_verse_head:
            return True
        elif isNumberOfWhiteSpacesLessThanTwo:
            return True
    else:
        return True


def is_verse_header(first_line_of_verse_lowercase):
    return '[' in first_line_of_verse_lowercase \
           or '(' in first_line_of_verse_lowercase \
           or 'verse' in first_line_of_verse_lowercase


def add_learning_tags_to_verse(verses):
    artist_verses_with_tags = []
    verse_with_tags = []
    for verse in verses:
        verse_first_line = verse[0]
        verse_with_tags.append("<verse_start>")
        if not is_verse_header(verse_first_line):
            verse_with_tags.append(verse_first_line + "<end_line>")

        for i in range(1, len(verse)):
            verse_with_tags.append(verse[i] + "<end_line>")
        artist_verses_with_tags.append(verse_with_tags)
        verse_with_tags = []

    return artist_verses_with_tags


def clear_artist_verses_of_background_voices_and_short_lines(artist_verses_with_tags):
    artist_cleaned_verses = []
    cleaned_verse = []
    for verse in artist_verses_with_tags:
        for i in range(1, len(verse) - 1):
            current_verse_line = verse[i]
            # delete all words between {}, () and [] with brackets and empty characters that surrounds brackets
            current_verse_line = re.sub(" ?[\(\[\{].*?[\)\]\}] ?", "", current_verse_line)

            number_of_words_in_line = len(re.findall(r'\w+', current_verse_line))
            if number_of_words_in_line > 2:
                cleaned_verse.append(current_verse_line)

        artist_cleaned_verses.append(cleaned_verse)
        cleaned_verse = []

    return artist_cleaned_verses


def make_text_lyrics_of_list_of_verses(artist_cleaned_verses):
    filtered_lyrics_of_song = ""
    for verse in artist_cleaned_verses:
        for line in verse:
            filtered_lyrics_of_song += line + '\n'
        filtered_lyrics_of_song += "\n"
    return filtered_lyrics_of_song


'''
MAIN
'''

artist_lyrics_folder_path = "./data/rap_lyrics_links/Fabolous/"
artist_lyrics_map = load_lyrics_into_map(artist_lyrics_folder_path)
artist_name = "Fabolous"

filtered_artist_lyrics_map = filter_artist_verses_and_return_map(artist_lyrics_map, artist_name)
