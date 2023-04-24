import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    #load data
    cases_train = pd.read_csv('../data/cases_2021_train.csv')
    cases_test = pd.read_csv('../data/cases_2021_test.csv')
    cases_location = pd.read_csv('../data/location_2021.csv')
    print(cases_train.groupby('outcome').size())

    #1.1
    #maps outcomes to four distinct outcomes, then use dictionary to find outcome_groups
    hospitalized = ['Discharged', 'Discharged from hospital', 'Hospitalized', 'critical condition', 'discharge', 'discharged']
    nonhospitalized = ['Alive', 'Receiving Treatment', 'Stable', 'Under treatment', 'recovering at home 03.03.2020', 'released from quarantine', 'stable', 'stable condition']
    deceased = ['Dead', 'Death', 'Deceased', 'Died', 'death', 'died']
    recovered = ['Recovered', 'recovered']
    four_values = ['hospitalized', 'nonhospitalized', 'deceased', 'recovered']
    four_conditions = [hospitalized, nonhospitalized, deceased, recovered]
    conditions = {}

    for i in range(4):
        for c in four_conditions[i]:
            conditions[c] = four_values[i]

    cases_train['outcome_group'] = cases_train['outcome'].apply(lambda x: conditions.get(x))
    cases_train.drop(['outcome'], axis=1, inplace=True)
    print()
    print(cases_train.groupby('outcome_group').size())

    '''
    #1.3
    #Binning into three distinct age group, then bar graph comparison
    valid = cases_train.dropna(axis=0, subset=['age'])
    valid = valid[valid['age'].str.isnumeric()]
    valid['age'] = valid['age'].apply(lambda x: float(x))
    age = valid.groupby('age').size()
    plt.title('Number of cases vs age')
    plt.xlabel('age')
    plt.ylabel('case(s)')
    plt.plot(age.index, age)
    plt.savefig('../plots/task-1.3/case_vs_age.png')
    plt.show()

    #pie
    #all
    plt.title('Cases_train outcome_group ratio')
    pie_outcome_ratio = cases_train.groupby('outcome_group').size()
    plt.pie(x=pie_outcome_ratio, labels=pie_outcome_ratio.index, autopct='%1.1f%%')
    plt.savefig('../plots/task-1.3/train_ratio.png')
    plt.show()

    #chronic
    plt.title('Chronic outcome_group ratio')
    pie_outcome_ratio = cases_train[cases_train['chronic_disease_binary']].groupby('outcome_group').size()
    plt.pie(x=pie_outcome_ratio, labels=pie_outcome_ratio.index, autopct='%1.1f%%')
    plt.savefig('../plots/task-1.3/chronic_ratio.png')
    plt.show()

    #non chronic
    plt.title('Non-chronic outcome_group ratio')
    pie_outcome_ratio = cases_train[cases_train['chronic_disease_binary'] == False].groupby('outcome_group').size()
    plt.pie(x=pie_outcome_ratio, labels=pie_outcome_ratio.index, autopct='%1.1f%%')
    plt.savefig('../plots/task-1.3/non-chronic_ratio.png')
    plt.show()

    #male
    plt.title('Male outcome_group ratio')
    pie_outcome_ratio = cases_train[cases_train['sex']=='male'].groupby('outcome_group').size()
    plt.pie(x=pie_outcome_ratio, labels=pie_outcome_ratio.index, autopct='%1.1f%%')
    plt.savefig('../plots/task-1.3/male_ratio.png')
    plt.show()

    #female
    plt.title('Female outcome_group ratio')
    pie_outcome_ratio = cases_train[cases_train['sex']=='female'].groupby('outcome_group').size()
    plt.pie(x=pie_outcome_ratio, labels=pie_outcome_ratio.index, autopct='%1.1f%%')
    plt.savefig('../plots/task-1.3/female_ratio.png')
    plt.show()

    #geo map
    #cases_train
    plt.title('Latitude vs longitude cases_train frequency')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.scatter(x=cases_train['latitude'], y=cases_train['longitude'])
    plt.savefig('../plots/task-1.3/train_map.png')
    plt.show()

    #cases_location
    plt.title('Latitude vs longitude cases_location frequency')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.scatter(x=cases_location['Lat'], y=cases_location['Long_'])
    plt.savefig('../plots/task-1.3/location_map.png')
    plt.show()

    #pie ratio of chronic, 4 chronic, 4 nonchronic
    #print numerical statistic
    print('~Train statistic~')
    print(cases_train.describe())
    print('\n~Test statistic~')
    print(cases_test.describe())
    print('\n~Location statistic~')
    print(cases_location.describe())
    #print nan values
    print('\n~Train NaN~')
    print(cases_train.isna().sum())
    print('\n~Test NaN~')
    print(cases_test.isna().sum())
    print('\n~Location NaN~')
    print(cases_location.isna().sum())
    '''

    #1.4
    cases_train.dropna(axis=0, subset=['age'], inplace=True)
    cases_test.dropna(axis=0, subset=['age'], inplace=True)

    #impute age
    #change format age-age to age by taking means, then floor age using int()
    def clean_age(x):
        age = str(x)
        age = age.split('-')
        
        if len(age) == 1 or age[1] == '':
            return int(float(age[0]))
        else:
            return int((float(age[1])-float(age[0]))/2)

    cases_train['age'] = cases_train['age'].apply(clean_age)
    cases_test['age'] = cases_test['age'].apply(clean_age)
    #cases_train.groupby('age').size()
    #cases_test.groupby('age').size()

    #drop nan 
    #only account for around 3% of dataset
    #imposible to impute reasonable values for sex and data_confirm
    #imposible to impute lat and long since it diffucult to pin point
    cases_train.dropna(axis=0, subset=['sex', 'date_confirmation'], inplace=True)
    cases_test.dropna(axis=0, subset=['sex', 'date_confirmation'], inplace=True)
    cases_location.dropna(axis=0, subset=['Lat', 'Long_', 'Incident_Rate'], inplace=True)

    #fill in default values of 'none' for missing additional_information and source
    cases_train[['additional_information', 'source']] = cases_train[['additional_information', 'source']].fillna(value='none')
    cases_test[['additional_information', 'source']] = cases_test[['additional_information', 'source']].fillna(value='none')

    #fill in default values of -1 for unknown recovered and active data (mostly US entries)
    cases_location[['Recovered', 'Active']] = cases_location[['Recovered', 'Active']].fillna(value=-1)

    #fill in default values of 'unspecified' for missing country or province (TW)
    cases_train[['province', 'country']] = cases_train[['province', 'country']].fillna(value='unspecified')
    cases_test[['province', 'country']] = cases_test[['province', 'country']].fillna(value='unspecified')
    cases_location[['Province_State', 'Country_Region']] = cases_location[['Province_State', 'Country_Region']].fillna(value='unspecified')

    #compute for missing Case_Fatality_Ratio, in the case 0/0 input 0
    cases_location = cases_location[cases_location['Deaths'] <= cases_location['Confirmed']]
    cases_location['Case_Fatality_Ratio'] = cases_location['Case_Fatality_Ratio'].fillna(cases_location['Deaths'] / cases_location['Confirmed'])
    cases_location['Case_Fatality_Ratio'].fillna(value=0, inplace=True)

    # #Testing print
    # print('Cases:')
    # print(len(cases_train.index))
    # print(len(cases_test.index))
    # print(len(cases_location.index))
    # #print nan values
    # print('\n~Train NaN~')
    # print(cases_train.isna().sum())
    # print('\n~Test NaN~')
    # print(cases_test.isna().sum())
    # print('\n~Location NaN~')
    # print(cases_location.isna().sum())

    #1.5
    #outlier for observed age
    #train
    age_lower_bound = 0
    mean = cases_train['age'].mean()
    std = cases_train['age'].std()
    age_upper_bound = mean + 3*std
    cases_train = cases_train[(age_lower_bound<=cases_train['age']) & (cases_train['age']<=age_upper_bound)]

    #test
    mean = cases_test['age'].mean()
    std = cases_test['age'].std()
    age_upper_bound = mean + 3*std
    cases_test = cases_test[(age_lower_bound<=cases_test['age']) & (cases_test['age']<=age_upper_bound)]

    #outlier for case_fatality_ratio
    ratio_lower_bound = 0
    # mean = cases_location['Case_Fatality_Ratio'].mean()
    # std = cases_location['Case_Fatality_Ratio'].std()
    # ratio_upper_bound = mean + 3*std
    cases_location = cases_location[ratio_lower_bound<=cases_location['Case_Fatality_Ratio']]

    #lat(-90,90) and long(-180,180) 
    cases_train = cases_train[(-90<=cases_train['latitude']) & (cases_train['latitude']<=90)]
    cases_train = cases_train[(-180<=cases_train['longitude']) & (cases_train['longitude']<=180)]
    cases_test = cases_test[(-90<=cases_test['latitude']) & (cases_test['latitude']<=90)]
    cases_test = cases_test[(-180<=cases_test['longitude']) & (cases_test['longitude']<=180)]
    cases_location = cases_location[(-90<=cases_location['Lat']) & (cases_location['Lat']<=90)]
    cases_location = cases_location[(-180<=cases_location['Long_']) & (cases_location['Long_']<=180)]
    #1.6
    #search for values that doesn't match
    # a = cases_train['country'].unique().tolist()
    # b = cases_test['country'].unique().tolist()
    # c = cases_location['Country_Region'].unique().tolist()

    # a_arr = []
    # for country in a:
    #     if country not in c:
    #         a_arr.append(country)
    # print(a_arr)
            
    # b_arr = []
    # for country in b:
    #     if country not in c:
    #         b_arr.append(country)
    # print(b_arr)

    #replace US; Korea, South
    cases_location = cases_location.replace('Korea, South' ,'South Korea')
    cases_location = cases_location.replace('US' ,'United States')

    rename_location = cases_location.rename(columns={'Country_Region': 'country', 'Province_State': 'province'})
    train_join_location = cases_train.merge(rename_location, how='left', on=['country', 'province'])
    test_join_location = cases_test.merge(rename_location, how='left', on=['country', 'province'])

    #drop nan from left join
    train_join_location.dropna(axis=0, inplace=True)
    test_join_location.dropna(axis=0, subset=['Combined_Key'], inplace=True)
    
    #print len of rows
    print()
    print('Train join location number of rows: ',len(train_join_location.index))
    print('Test join location number of rows: ',len(test_join_location.index))

    #1.7
    #cases_location
    feature_train = train_join_location[['age', 'sex', 'province', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio', 'outcome_group']]
    feature_test = test_join_location[['age', 'sex', 'province', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio', 'outcome_group']]
    train_join_location.to_csv('../results/cases_2021_train_processed.csv')
    test_join_location.to_csv('../results/cases_2021_test_processed.csv')
    feature_train.to_csv('../results/cases_2021_train_processed_features.csv')
    feature_test.to_csv('../results/cases_2021_test_processed_features.csv')
    cases_location.to_csv('../results/location_2021_processed.csv')

if __name__ == '__main__':
    main()

