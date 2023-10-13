import random


class Questions:
    qs = []

    def __init__(self, path_to_questions):
        file = open(path_to_questions, 'r')

        for line in file.readlines():
            self.qs.append(line.rstrip('\n'))

    def random(self) -> str:  # random questions
        return random.choice(self.qs)

    def random_rm(self) -> str:  # random questions and remove it from possible questions
        q = random.choice(self.qs)
        self.qs.remove(q)
        return q