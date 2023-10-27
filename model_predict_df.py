import os
import pandas as pd
import numpy as np
import joblib
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator
from pymatgen.analysis.cost import CostAnalyzer, CostDBElements

def get_cost(comp_list):
    ca = CostAnalyzer(costdb=CostDBElements())
    costs = list()
    for comp in comp_list:
        cost = ca.get_cost_per_kg(comp=comp)
        costs.append(cost)
    return costs

def get_stability(df_test):
    d = 'Stability_model'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))

    features = df_features.columns.tolist()
    df_test2 = df_test[features]
    X_stab = scaler.transform(df_test2)
    stabilities = model.predict(X_stab)

    return stabilities

def get_barrier(df_test):
    d = 'Barrier_model'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))

    features = df_features.columns.tolist()
    X_barrier = df_test[features]
    X_barrier = scaler.transform(X_barrier)
    barriers = model.predict(X_barrier)

    return barriers

def get_asr(df_test):
    d = 'ASR_model'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))

    features = df_features.columns.tolist()
    df_test = df_test[features]

    X_ASR = scaler.transform(df_test)

    asrs = model.predict(X_ASR)

    # Get ebars and recalibrate them
    a = 0.42824232546669644
    b = 0.36341790743237223
    errs_list = list()
    for i, x in X_ASR.iterrows():
        preds_list = list()
        for pred in model.model.estimators_:
            preds_list.append(pred.predict(np.array(x).reshape(1, -1))[0])
        errs_list.append(np.std(preds_list))
    asr_ebars = a * np.array(errs_list) + b

    return asrs, asr_ebars

def process_data(comp_list, elec_list):
    X = pd.DataFrame(np.empty((len(comp_list),)))
    y = pd.DataFrame(np.empty((len(comp_list),)))

    df_test = pd.DataFrame({'Material composition': comp_list})

    # Try this both ways depending on mastml version used.
    try:
        X, y = ElementalFeatureGenerator(composition_df=df_test['Material composition'],
                                    feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min','difference'],
                                    remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)
    except:
        X, y = ElementalFeatureGenerator(featurize_df=df_test['Material composition'],
                                         feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min',
                                                        'difference'], remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)

    df_test = pd.concat([df_test, X], axis=1)

    elec_cls_0 = list()
    elec_cls_1 = list()
    elec_cls_2 = list()
    elec_cls_3 = list()
    for elec in elec_list:
        if elec == 'ceria':
            elec_cls_0.append(1)
            elec_cls_1.append(0)
            elec_cls_2.append(0)
            elec_cls_3.append(0)
        elif elec == 'mixed':
            elec_cls_0.append(0)
            elec_cls_1.append(1)
            elec_cls_2.append(0)
            elec_cls_3.append(0)
        elif elec == 'perovskite':
            elec_cls_0.append(0)
            elec_cls_1.append(0)
            elec_cls_2.append(1)
            elec_cls_3.append(0)
        elif elec == 'zirconia':
            elec_cls_0.append(0)
            elec_cls_1.append(0)
            elec_cls_2.append(0)
            elec_cls_3.append(1)
        else:
            raise ValueError('Invalid electrolyte choice detected. Valid choices are "ceria", "mixed", "perovskite", "zirconia"')

    df_test['Electrolyte class_0'] = elec_cls_0  # ceria
    df_test['Electrolyte class_1'] = elec_cls_1  # mixed
    df_test['Electrolyte class_2'] = elec_cls_2  # perovskite
    df_test['Electrolyte class_3'] = elec_cls_3  # zirconia

    return df_test

def make_predictions(comp_list, elec_list):

    # Check comp and elec list lengths match
    assert len(comp_list) == len(elec_list)

    # Process data
    df_test = process_data(comp_list, elec_list)

    # Calculate the cost of the materials
    costs = get_cost(comp_list)

    # Get the ML-predicted stability of the materials
    stabilities = get_stability(df_test)

    # Get the ML-predicted ASR barrier of the materials
    barriers = get_barrier(df_test)

    df_test['ML pred ASR barrier (eV)'] = barriers
    asrs, asr_ebars = get_asr(df_test)

    pred_dict = {'Compositions': comp_list,
                 'Electrolytes': elec_list,
                 'Cost ($/kg)': costs,
                 'Stability @ 500C (meV/atom)': stabilities,
                 'ASR barrier (eV)': barriers,
                 'log ASR at 500C (Ohm-cm2)': asrs,
                 'log ASR error (Ohm-cm2)': asr_ebars}

    return pd.DataFrame(pred_dict)

