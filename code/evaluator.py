import csv

test_number = 0
test_match = 0

toxic_number = 0
toxic_test = 0
toxic_match = 0

severe_toxic_number = 0
severe_toxic_test = 0
severe_toxic_match = 0

obscene_number = 0
obscene_test = 0
obscene_match = 0

threat_number = 0
threat_test = 0
threat_match = 0

insult_number = 0
insult_test = 0
insult_match = 0

identity_hate_number = 0
identity_hate_test = 0
identity_hate_match = 0

non_offensive = 0
non_offensive_test = 0

available = dict()
temp = []

if __name__ == '__main__':

    test_label = open("test_labels.csv")
    csv_reader = csv.reader(test_label)
    for data in csv_reader:
        if data[1] != "-1" and data[1] != "toxic":
            available[data[0]] = data[1:]
            test_number += 1
            non_offensive_flag = True
            if data[1] == "1":
                toxic_number += 1
                non_offensive_flag = False
            if data[2] == "1":
                severe_toxic_number += 1
                non_offensive_flag = False
            if data[3] == "1":
                obscene_number += 1
                non_offensive_flag = False
            if data[4] == "1":
                threat_number += 1
                non_offensive_flag = False
            if data[5] == "1":
                insult_number += 1
                non_offensive_flag = False
            if data[6] == "1":
                identity_hate_number += 1
                non_offensive_flag = False
            if non_offensive_flag:
                non_offensive += 1
            temp = available.get(data[0])
            temp.append(non_offensive_flag)
            available[data[0]] = temp

    # print(available)

    results = open("new_res.csv")
    result_reader = csv.reader(results)

    def classifier(res):
        if float(res) > 0.5:
            return int(1)
        return int(0)

    for result in result_reader:
        if result[0] in available.keys():
            non_offensive_test = True

            label = available.get(result[0])
            # print(label)

            toxic_result = classifier(result[1])
            toxic_label = int(label[0])
            if toxic_result == toxic_label and toxic_result == 1:
                toxic_match += 1
            if toxic_result == 1:
                non_offensive_test = False
                toxic_test += 1

            severe_toxic_result = classifier(result[2])
            severe_toxic_label = int(label[1])
            if severe_toxic_result == severe_toxic_label and severe_toxic_result == 1:
                severe_toxic_match += 1
            if severe_toxic_result == 1:
                non_offensive_test = False
                severe_toxic_test += 1

            obscene_result = classifier(result[3])
            obscene_label = int(label[2])
            if obscene_result == obscene_label and obscene_result == 1:
                obscene_match += 1
            if obscene_result == 1:
                non_offensive_test = False
                obscene_test += 1

            threat_result = classifier(result[4])
            threat_label = int(label[3])
            if threat_result == threat_label and threat_result == 1:
                threat_match += 1
            if threat_result == 1:
                non_offensive_test = False
                threat_test += 1

            insult_result = classifier(result[5])
            insult_label = int(label[4])
            if insult_result == insult_label and insult_result == 1:
                insult_match += 1
            if insult_result == 1:
                non_offensive_test = False
                insult_test += 1

            identity_hate_label = int(label[5])

            identity_hate_result = classifier(result[6])

            if identity_hate_result == identity_hate_label and identity_hate_result == 1:
                identity_hate_match += 1
            if identity_hate_result == 1:
                non_offensive_test = False
                identity_hate_test += 1

            if non_offensive_test == label[6]:
                test_match += 1
            if non_offensive_test:
                non_offensive_test += 1

            available.pop(result[0])

    print("test number:")
    print(test_number)
    print("test match:")
    print(test_match)
    print()
    print("toxic number:")
    print(toxic_number)
    print(test_number-toxic_number)
    print("toxic match:")
    print(toxic_match)
    print("toxic test:")
    print(toxic_test)
    print(test_number - toxic_test)
    print()
    print("severe_toxic_number")
    print(severe_toxic_number)
    print(test_number-severe_toxic_number)
    print("severe_toxic_match")
    print(severe_toxic_match)
    print("severe_toxic_test")
    print(severe_toxic_test)
    print(test_number-severe_toxic_test)
    print()
    print("obscene number")
    print(obscene_number)
    print(test_number-obscene_number)
    print("obscene_match")
    print(obscene_match)
    print("obscene_test")
    print(obscene_test)
    print(test_number - obscene_test)
    print()
    print("threat number")
    print(threat_number)
    print(test_number - threat_number)
    print("threat_match")
    print(threat_match)
    print("threat_test")
    print(threat_test)
    print(test_number-threat_test)
    print()
    print("insult number")
    print(insult_number)
    print(test_number-insult_number)
    print("insult_match")
    print(insult_match)
    print("insult_test")
    print(insult_test)
    print(test_number-insult_test)
    print()
    print("hate number")
    print(identity_hate_number)
    print(test_number-identity_hate_number)
    print("hate match")
    print(identity_hate_match)
    print("hate_test")
    print(identity_hate_test)
    print(test_number-identity_hate_test)
