import re
import pronouncing as pr


def parse_cmu_dict():
    pronunciations = dict()
    for line in open("cmudict-0.7b", encoding="ISO-8859-1").readlines():
        line = line.strip()
        if line.startswith(';'): continue
        word, phones = line.split("  ")
        word = word.rstrip("(0123)").lower()
        if word not in pronunciations:
            pronunciations[word] = []
        pronunciations[word].append(phones)

    return pronunciations


def count_syllables(phones):
    return sum([phones.count(s) for s in '012'])


def rhyming_part(phones):
    phones_list = phones.split()
    for i in range(len(phones_list) - 1, 0, -1):
        if phones_list[i][-1] in '12':
            return ' '.join(phones_list[i:])


'''

    Ukoliko mi zatreba mogu iskoristiti neke od ovih funkcionalnosti biblioteke pronouncing
    pr.search(phones)  -> kada unesemo neki fonetski izraz ovo nam vraca sve reci koje imaju
                            taj sablon fonema
                            
    pr.stresses(phones) -> kada unesemo neki fonetski izraz ovo nam vraca sablon stresslevela svakog
                            samoglasnika 
                            
    pr.search_stresses(r"^10000$") -> na osnovu sablona stresslvla samoglasnika vraca reci koje se poklapaju tome

'''