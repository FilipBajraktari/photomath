#len of start = 25
start = '<annotation type="truth">'
#len of end = 13
end = '</annotation>'

dir = '../x_n.inkml'
def Truth(dir):
    with open(dir, 'r') as file:
        for line in file:

            stripped_line = line.strip()
            if stripped_line[:25]==start and stripped_line[-13:]:
                return stripped_line[25:-13]

if __name__ == '__main__':
    print(Truth(dir))