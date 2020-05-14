import os
import io


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


artist_lyrics_folder_path = "./data/rap_lyrics_links/Drake/"
artist_lyrics_map = load_lyrics_into_map(artist_lyrics_folder_path)

for lyrics_filename, lyrics_text in artist_lyrics_map.items():
    print(lyrics_filename)
    print("=" * 80)
    lyrics_text_as_buffer = io.StringIO(lyrics_text)
    lyrics_lines = lyrics_text_as_buffer.readlines()
    test_text = ""
    for line in lyrics_lines:
        if line.endswith("\n"):
            line = line.replace("\n", "")
        line += "<end_line>\n"
        test_text += line
    print(test_text)


