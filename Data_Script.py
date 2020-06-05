import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data base rules:
# 97 not appliable
# 2 no
# 1 yes


data = pd.read_csv("DataCovid2020.csv", encoding='latin-1')  # We read the file
file = open("results.txt", "w")  # We create a file to save all the information gather
file.write("Data Covid19 Mexico \n")

# Function to all the information about age
def general():
    global mean, yonger, older, std, count, UCI
    file.write("General data\n")
    mean = data["EDAD"].describe()["mean"]
    file.write("Mean of age: " + str(mean) +"\n")
    yonger = data["EDAD"].describe()["min"]
    file.write("Younger Pacient: " + str(yonger) + "\n")
    older = data["EDAD"].describe()["max"]
    file.write("Older Pacient: " + str(older) + "\n")
    std = data["EDAD"].describe()["std"]
    file.write("Standar Derivation Age: " + str(std) + "\n")
    count = data["EDAD"].describe()["count"]
    file.write("Infected of covid19 in Mexico: " + str(count) + "\n")
    UCI = data.loc[data["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Pacients that needed Intensive Care: " + str(UCI) + "\n")

# Function of data that the patients are female
def female():
    global countFemale, meanFemale, countPregnant, pregnantUCI, asmaFemaleCount, asmaFemaleUCI
    global asmaFemaleNeu, diabetesFemaleCount, diabetesFemaleUCI, countbpFemale, bpFemaleUCI
    global countoFemale, oFemaleUCI, countindFemale, indFemaleUCI

    # General Data
    file.write("General data Female\n")
    female = data.loc[data["SEXO"] == 1]  # if the variable sexo is 1 then the patient is a female
    countFemale = female["SEXO"].describe()["count"]
    meanFemale = female["EDAD"].describe()["mean"]
    file.write("Total Female Infected of covid19: " + str(countFemale) + "\n")
    file.write("Mean Female: " + str(meanFemale) + "\n")

    # Female pregnant
    file.write("Pregnant data Female\n")
    pregnant = female.loc[female["EMBARAZO"] == 1]
    countPregnant = pregnant["EMBARAZO"].describe()["count"]
    pregnantUCI = pregnant.loc[pregnant["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Female Pregnant Infected of covid19: " + str(countPregnant) + "\n")
    file.write("Female Pregnant that needed Intensive Care: " + str(pregnantUCI) + "\n")

    # Female with asma
    asmaFemale = female.loc[female["ASMA"] == 1]
    asmaFemaleCount = asmaFemale["ASMA"].describe()["count"]
    asmaFemaleUCI = asmaFemale.loc[asmaFemale["UCI"] == 1]["UCI"].describe()["count"]
    asmaFemaleNeu = asmaFemale.loc[asmaFemale["NEUMONIA"] == 1]["NEUMONIA"].describe()["count"]
    file.write("Total Female Asma Infected of covid19: " + str(asmaFemaleCount) + "\n")
    file.write("Female Asma that needed Intensive Care: " + str(asmaFemaleUCI) + "\n")
    file.write("Female Asma that develop Neumonia: " + str(asmaFemaleNeu) + "\n")

    # Female with diabetes
    diabetesFemale = female.loc[female["DIABETES"] == 1]
    diabetesFemaleCount = diabetesFemale["DIABETES"].describe()["count"]
    diabetesFemaleUCI = diabetesFemale.loc[diabetesFemale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Female diabetes Infected of covid19: " + str(diabetesFemaleCount) + "\n")
    file.write("Female diabetes that needed Intensive Care: " + str(diabetesFemaleUCI) + "\n")

    # Female with high blood pressure
    bpFemale = female.loc[female["HIPERTENSION"] == 1]
    countbpFemale = bpFemale["HIPERTENSION"].describe()["count"]
    bpFemaleUCI = bpFemale.loc[bpFemale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Female high blood pressure Infected of covid19: " + str(countbpFemale) + "\n")
    file.write("Female high blood pressure that needed Intensive Care: " + str(bpFemaleUCI) + "\n")

    # Female with OBESITY
    oFemale = female.loc[female["OBESIDAD"] == 1]
    countoFemale = oFemale["OBESIDAD"].describe()["count"]
    oFemaleUCI = oFemale.loc[oFemale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Female obesity Infected of covid19: " + str(countoFemale) + "\n")
    file.write("Female obesity that needed Intensive Care: " + str(oFemaleUCI) + "\n")

    # Female indigenous
    indFemale = female.loc[female["HABLA_LENGUA_INDIG"] == 1]
    countindFemale = indFemale["HABLA_LENGUA_INDIG"].describe()["count"]
    indFemaleUCI = indFemale.loc[indFemale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Female indigenous  Infected of covid19: " + str(countindFemale) + "\n")
    file.write("Female indigenous that needed Intensive Care: " + str(indFemaleUCI) + "\n")


# Function of data that the patients are female
def male():
    global countMale,meanMale,asmaMaleCount,asmaMaleUCI,asmaMaleNeu,diabetesMaleCount
    global diabetesMaleUCI,countbpMale,bpMaleUCI,countoMale,oMaleUCI,countindMale,indMaleUCI
    # General Data
    male = data.loc[data["SEXO"] == 2]  # if the variable sexo is 2 is male
    countMale = male["SEXO"].describe()["count"]
    meanMale = male["EDAD"].describe()["mean"]
    file.write("Total Male Infected of covid19: " + str(countMale) + "\n")
    file.write("Mean Male: " + str(meanMale) + "\n")

    # Male with asma
    asmaMale = male.loc[male["ASMA"] == 1]
    asmaMaleCount = asmaMale["ASMA"].describe()["count"]
    asmaMaleUCI = asmaMale.loc[asmaMale["UCI"] == 1]["UCI"].describe()["count"]
    asmaMaleNeu = asmaMale.loc[asmaMale["NEUMONIA"] == 1]["NEUMONIA"].describe()["count"]
    file.write("Total Male Asma Infected of covid19: " + str(asmaMaleCount) + "\n")
    file.write("Male Asma that needed Intensive Care: " + str(asmaMaleUCI) + "\n")
    file.write("Male Asma that develop Neumonia: " + str(asmaMaleNeu) + "\n")

    # Male with diabetes
    diabetesMale = male.loc[male["DIABETES"] == 1]
    diabetesMaleCount = diabetesMale["DIABETES"].describe()["count"]
    diabetesMaleUCI = diabetesMale.loc[diabetesMale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Male diabetes Infected of covid19: " + str(diabetesMaleCount) + "\n")
    file.write("Male diabetes that needed Intensive Care: " + str(diabetesMaleUCI) + "\n")

    # Male with high blood pressure
    bpMale = male.loc[male["HIPERTENSION"] == 1]
    countbpMale = bpMale["HIPERTENSION"].describe()["count"]
    bpMaleUCI = bpMale.loc[bpMale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Male high blood pressure Infected of covid19: " + str(countbpMale) + "\n")
    file.write("Male high blood pressure that needed Intensive Care: " + str(bpMaleUCI) + "\n")

    # Male with OBESITY
    oMale = male.loc[male["OBESIDAD"] == 1]
    countoMale = oMale["OBESIDAD"].describe()["count"]
    oMaleUCI = oMale.loc[oMale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Male obesity Infected of covid19: " + str(countoMale) + "\n")
    file.write("Male obesity that needed Intensive Care: " + str(oMaleUCI) + "\n")

    # Male indigenous
    indMale = male.loc[male["HABLA_LENGUA_INDIG"] == 1]
    countindMale = indMale["HABLA_LENGUA_INDIG"].describe()["count"]
    indMaleUCI = indMale.loc[indMale["UCI"] == 1]["UCI"].describe()["count"]
    file.write("Total Male indigenous  Infected of covid19: " + str(countindMale) + "\n")
    file.write("Male indigenous that needed Intensive Care: " + str(indMaleUCI) + "\n")

general()
female()
male()

# Graphs codes

# Graph comparing female and male data
labels = ['Pregnant', 'Asma', 'Diabetes', 'high blood pressure', 'Obesity', "Indigenous"]
men = [0,asmaMaleCount, diabetesMaleCount, countbpMale, countoMale, countindMale]
women = [countPregnant, asmaFemaleCount, diabetesFemaleCount, countbpFemale, countoFemale, countindFemale]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, men, width, label='Men')
rects2 = ax.bar(x + width / 2, women, width, label='Women')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

# Graph that represents the age
np.random.seed(19680801)
# example data
mu = mean  # mean of distribution
sigma = std  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)
num_bins = 50
fig, ax = plt.subplots()
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)
ax.set_xlabel('Ages')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram Pacient Ages')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

# Female Graph
labels = ['Pregnant', 'Asma', 'Diabetes', 'high blood pressure', 'Obesity', "Indigenous"]
IC = [pregnantUCI,asmaFemaleUCI, diabetesFemaleUCI, bpFemaleUCI, oFemaleUCI, indFemaleUCI]
patients = [countPregnant, asmaFemaleCount, diabetesFemaleCount, countbpFemale, countoFemale, countindFemale]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, IC, width, label='Intensive Care')
rects2 = ax.bar(x + width / 2, patients, width, label='Total of patients')
ax.set_ylabel('Number of patients')
ax.set_title('Comparative between female patients that needed Intensive Care')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

# Male Graph
labels = ['Asma', 'Diabetes', 'high blood pressure', 'Obesity', "Indigenous"]
IC = [asmaMaleUCI, diabetesMaleUCI, bpMaleUCI, oMaleUCI, indMaleUCI]
patients = [asmaMaleCount, diabetesMaleCount, countbpMale, countoMale, countindMale]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, IC, width, label='Intensive Care')
rects2 = ax.bar(x + width / 2, patients, width, label='Total of patients')
ax.set_ylabel('Number of patients')
ax.set_title('Comparative between male patients that needed Intensive Care')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

# Graph general
labels = 'Male', 'Female'
sizes = [(countMale/count)*100, (countFemale/count)*100]
explode = (0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()