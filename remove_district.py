import csv
import operator

state_set = set(["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","Province_State"])


count = 0

with open('/Users/hangl/Desktop/CS145/data/combined_csv.csv', 'rt') as temp, open('/Users/hangl/Desktop/CS145/data/sept_final.csv', 'wt') as out:
    inp = csv.reader(temp)
    writer = csv.writer(out)
    for row in inp:
        writer.writerow(row)
        break
    sort = sorted(inp, key=operator.itemgetter(2))
    for row in sort:
        if row[0] in state_set:
            writer.writerow(row)
            count += 1

print(count)
out.close()
