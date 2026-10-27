import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
BLUE, RED, SKY = '#17365D', '#C62828', '#83B4D8'

st.set_page_config(page_title='Explainable Credit Risk | Research Dashboard', page_icon='📊', layout='wide')
st.markdown('''<style>
.stApp {background:#F6F8FC;} .block-container {max-width:1430px;padding-top:1.5rem;}
h1,h2,h3,h4,label {font-weight:750 !important;color:#17365D !important;}
[data-testid="stSidebar"] {background:#EAF2FA;border-right:1px solid #D7E3EE;}
[data-testid="stMetric"] {background:white;border:1px solid #DCE5EF;border-radius:12px;padding:14px;}
[data-testid="stMetricValue"] {font-weight:800;color:#17365D;}
[data-testid="stExpander"] {background:white;border:1px solid #DDE5EF;border-radius:12px;margin-bottom:10px;}
[data-testid="stExpander"] summary {font-weight:750;color:#17365D;}
div.stFormSubmitButton > button[kind="primary"], div.stButton > button[kind="primary"] {background:#C62828;color:white;border:0;border-radius:9px;font-weight:800;min-height:44px;}
div.stFormSubmitButton > button[kind="primary"]:hover, div.stButton > button[kind="primary"]:hover {background:#A51C1C;color:white;}
.stTabs [data-baseweb="tab"] {font-weight:750;}
.hero {background:linear-gradient(110deg,#17365D,#285C8A);padding:23px 26px;border-radius:14px;color:white;margin-bottom:14px;}
.hero h1 {color:white !important;font-size:1.8rem;margin:0 0 5px;}.hero p {color:#E9F2FA;font-size:.96rem;margin:0;}
.feature-card {border-radius:12px;background:white;border:1px solid #DDE5EF;padding:14px 16px;min-height:105px;}
.feature-card .big {font-weight:850;font-size:1.65rem;color:#17365D;}.feature-card .label {font-weight:800;font-size:.97rem;color:#17365D;}
.feature-card .small {font-size:.82rem;color:#516176;line-height:1.5;}
.info-box {background:white;border:1px solid #DFE7F0;padding:16px;border-radius:12px;margin-bottom:12px;}
.info-title {color:#17365D;font-weight:800;margin-bottom:5px;}.info-copy {color:#45566C;line-height:1.55;font-size:.86rem;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(BASE/'model.joblib'), joblib.load(BASE/'encoder.joblib')

@st.cache_data
def load_config():
    return json.loads((BASE/'config.json').read_text(encoding='utf-8'))

model, encoder = load_model()
config = load_config()
fields = config['features']
order = config['feature_order']
classes = [str(x) for x in encoder.classes_]
POS = str(config['positive_class'])
THRESHOLD = float(config['threshold'])
assert len(order) == 20 and POS in classes and abs(THRESHOLD-0.30) < 1e-9, 'Unexpected credit model metadata'

# German Credit (Statlog) attributes, with original category codes retained.
DESCRIPTIONS = {
'Attribute1':'Checking Account Status', 'Attribute2':'Credit Duration (Months)',
'Attribute3':'Credit History', 'Attribute4':'Credit Purpose',
'Attribute5':'Credit Amount', 'Attribute6':'Savings / Bonds',
'Attribute7':'Employment Duration', 'Attribute8':'Installment Rate (%)',
'Attribute9':'Personal Status / Sex (Combined Code)',
'Attribute10':'Other Debtors / Guarantors',
'Attribute11':'Present Residence (Years)', 'Attribute12':'Property',
'Attribute13':'Age (Years)', 'Attribute14':'Other Installment Plans',
'Attribute15':'Housing', 'Attribute16':'Existing Credits',
'Attribute17':'Occupation', 'Attribute18':'Dependents',
'Attribute19':'Telephone', 'Attribute20':'Foreign Worker'
}
CODE_LABELS = {
'A11':'Below zero balance', 'A12':'Balance 0 to below 200', 'A13':'Balance 200 or above', 'A14':'No checking account',
'A30':'No credits / credits repaid duly', 'A31':'All credits duly repaid at this bank',
'A32':'Existing credits repaid duly', 'A33':'Delay in past payment', 'A34':'Critical account / other credits',
'A40':'Car (new)', 'A41':'Car (used)', 'A42':'Furniture/equipment', 'A43':'Radio/television',
'A44':'Domestic appliances', 'A45':'Repairs', 'A46':'Education', 'A47':'Vacation (code)',
'A48':'Retraining', 'A49':'Business', 'A410':'Other purposes',
'A61':'Savings below 100', 'A62':'Savings 100–499', 'A63':'Savings 500–999', 'A64':'Savings 1000+', 'A65':'Unknown/no savings account',
'A71':'Unemployed', 'A72':'Employment below 1 year', 'A73':'Employment 1–3 years',
'A74':'Employment 4–6 years', 'A75':'Employment 7 years or more',
'A91':'Male divorced/separated', 'A92':'Female divorced/separated/married',
'A93':'Male single', 'A94':'Male married/widowed', 'A95':'Female single',
'A101':'None', 'A102':'Co-applicant', 'A103':'Guarantor',
'A121':'Real estate', 'A122':'Building society/life insurance', 'A123':'Car/other property', 'A124':'Unknown/no property',
'A141':'Bank', 'A142':'Stores', 'A143':'None',
'A151':'Rent', 'A152':'Own', 'A153':'For free',
'A171':'Unemployed/unskilled non-resident', 'A172':'Unskilled resident',
'A173':'Skilled employee/official', 'A174':'Management/self-employed/highly qualified',
'A191':'No registered telephone', 'A192':'Registered telephone',
'A201':'Yes (foreign worker)', 'A202':'No (foreign worker)'
}
GROUPS = [
('01 · Financial Profile', ['Attribute1','Attribute3','Attribute5','Attribute6','Attribute8','Attribute14','Attribute16']),
('02 · Credit Purpose and Commitments', ['Attribute2','Attribute4','Attribute10','Attribute11','Attribute12','Attribute15']),
('03 · Employment and Applicant Details', ['Attribute7','Attribute9','Attribute13','Attribute17','Attribute18','Attribute19','Attribute20'])
]


def convert_value(value, field):
    if pd.isna(value): return np.nan
    kind = field['data_type']
    if kind == 'integer':
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"{DESCRIPTIONS.get(field['name'],field['name'])} requires a whole number")
        return int(number)
    if kind == 'float':
        number = float(value)
        if not np.isfinite(number): raise ValueError(f"Invalid number for {field['name']}")
        return number
    return str(value)


def prepare_input(df):
    missing = [c for c in order if c not in df.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    result = df[order].copy()
    for field in fields:
        key = field['name']
        result[key] = result[key].map(lambda v, f=field: convert_value(v,f))
        if field['input_type'] == 'select':
            allowed = {str(v) for v in field['choices']}
            present = {str(v) for v in result[key].dropna().unique()}
            extra = present - allowed
            if extra and field['data_type'] == 'string':
                raise ValueError(f'Unsupported code(s) for {DESCRIPTIONS.get(key,key)}: {sorted(extra)}')
    return result


def predict(df):
    probs = model.predict_proba(prepare_input(df))
    idx = classes.index(POS)
    label_idx = np.where(probs[:,idx] >= THRESHOLD, idx, 1-idx)
    output = pd.DataFrame({'Prediction': encoder.inverse_transform(label_idx)})
    output['Decision_Label'] = np.where(label_idx == idx, 'Elevated bad-credit prediction', 'Lower bad-credit prediction')
    for i,c in enumerate(classes): output[f'Probability_{c}'] = probs[:,i]
    return output


def chart(probabilities):
    names = ['Good credit' if c == '1' else 'Bad credit' for c in classes]
    values = np.asarray(probabilities,dtype=float)*100
    fig,ax = plt.subplots(figsize=(5.9,1.85))
    bars=ax.barh(names,values,color=[BLUE if c=='1' else RED for c in classes],height=.44)
    ax.invert_yaxis(); ax.set_xlim(0,110)
    ax.set_xticks([0,25,50,75,100]); ax.set_xlabel('Model-estimated probability (%)',fontweight='bold')
    for bar,val in zip(bars,values):
        ax.text(min(val+2,103),bar.get_y()+bar.get_height()/2,f'{val:.1f}%',va='center',fontweight='bold',color=BLUE,fontsize=9)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight('bold')
    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    for edge in ['top','right','left']: ax.spines[edge].set_visible(False)
    fig.tight_layout(); return fig

with st.sidebar:
    st.markdown('## 📊 Credit Risk AI')
    st.caption('Explainable research prototype')
    st.divider()
    st.markdown('**Model details**')
    st.write('**Classifier:** XGBoost')
    st.write('**Input features:** 20')
    st.write('**Positive class:** Bad credit (2)')
    st.write('**Frozen threshold:** 0.30')
    st.divider()
    st.warning('Research demonstration only. Do not use for credit approval or denial.')

st.markdown('''<div class="hero"><h1>📊 Explainable Credit Risk Assessment</h1>
<p>Fairness-aware machine learning research · German Credit (Statlog) benchmark</p></div>''',unsafe_allow_html=True)
a,b,c=st.columns(3)
with a: st.markdown('<div class="feature-card"><div class="label">🏦 Credit profile</div><div class="big">20 attributes</div><div class="small">Financial, credit and applicant information</div></div>',unsafe_allow_html=True)
with b: st.markdown('<div class="feature-card"><div class="label">🧠 Predictive model</div><div class="big">XGBoost</div><div class="small">Saved, frozen preprocessing pipeline</div></div>',unsafe_allow_html=True)
with c: st.markdown('<div class="feature-card"><div class="label">🎯 Decision threshold</div><div class="big" style="color:#C62828">0.30</div><div class="small">Used for bad-credit class prediction</div></div>',unsafe_allow_html=True)
st.caption('Historic benchmark model. Scores are estimates, not creditworthiness decisions or fairness guarantees.')

tab1,tab2,tab3=st.tabs(['🎯 Individual Assessment','📁 Batch CSV Assessment','📊 Model Explainability'])
with tab1:
    main,aside=st.columns([3.15,1.2],gap='large')
    with main:
        st.markdown('### Applicant and Credit Information')
        st.caption('All 20 original model inputs are preserved. Category descriptions are provided for convenience.')
        vals={}
        field_lookup={f['name']:f for f in fields}
        with st.form('credit_prediction',clear_on_submit=False):
            for j,(section,names) in enumerate(GROUPS):
                with st.expander(section,expanded=(j==0)):
                    cols=st.columns(3,gap='medium')
                    for i,key in enumerate(names):
                        f=field_lookup[key]
                        with cols[i%3]:
                            label=DESCRIPTIONS.get(key,key)
                            if f['input_type']=='select':
                                options=f['choices']
                                try: default_index=options.index(f['default'])
                                except ValueError: default_index=0
                                vals[key]=st.selectbox(label,options,index=default_index,
                                    format_func=lambda v:f'{v} — {CODE_LABELS.get(str(v),"Dataset category")}',
                                    key='credit_'+key,help='Original dataset field: '+key)
                            elif f['data_type']=='integer':
                                vals[key]=st.number_input(label,value=int(f['default']),step=1,key='credit_'+key)
                            else:
                                vals[key]=st.number_input(label,value=float(f['default']),step=.1,format='%.3f',key='credit_'+key)
            submit=st.form_submit_button('⚡ Generate Credit Risk Prediction',type='primary',use_container_width=True)
        if submit:
            try:
                result=predict(pd.DataFrame([vals]))
                st.session_state['credit_result']={
                    'label':str(result.iloc[0]['Decision_Label']),
                    'probabilities':[float(result.iloc[0][f'Probability_{cl}']) for cl in classes]
                }
            except Exception as e: st.error(f'Prediction failed: {e}')
        if 'credit_result' in st.session_state:
            r=st.session_state['credit_result']; probs=np.asarray(r['probabilities'],dtype=float)
            risk=probs[classes.index(POS)]
            st.markdown('### Assessment Results')
            k1,k2,k3=st.columns(3)
            k1.metric('Model Classification',r['label'])
            k2.metric('Bad-credit probability',f'{risk*100:.2f}%')
            k3.metric('Decision threshold',f'{THRESHOLD:.2f}')
            if risk >= THRESHOLD:
                st.error('Model flags an elevated bad-credit prediction. This is NOT a lending decision.')
            else:
                st.info('Model predicts the lower bad-credit category. This is NOT a lending recommendation.')
            fig=chart(probs); st.pyplot(fig,use_container_width=False); plt.close(fig)
            st.caption('Model probabilities may be uncalibrated. The benchmark does not establish individual lending eligibility.')
    with aside:
        st.markdown('<div class="info-box"><div class="info-title">ℹ️ About this prototype</div><div class="info-copy">Research demonstration for predicting historical good/bad credit labels. The dataset contains 1,000 records, and the model uses 20 attributes.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box"><div class="info-title">🔎 Interpretable inputs</div><div class="info-copy">Original UCI category codes are retained for model compatibility. Human-readable descriptions are displayed alongside the codes.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box"><div class="info-title">⚖️ Fairness limitations</div><div class="info-copy">Historical group disparities were examined in research. A risk score does not establish a fair outcome, and this tool must not be used to make real loan decisions.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box"><div class="info-title">🧪 Evaluation context</div><div class="info-copy">Frozen XGBoost model; evaluated using a held-out partition. No parameters are updated by user input.</div></div>',unsafe_allow_html=True)

with tab2:
    st.markdown('### Batch Credit Risk Assessment')
    st.write('Upload a CSV containing exactly the 20 required feature columns. Original dataset category codes must be used.')
    template=pd.DataFrame([{f['name']:f['default'] for f in fields}])
    st.download_button('⬇ Download CSV Input Template',template.to_csv(index=False),file_name='credit_risk_input_template.csv',mime='text/csv')
    uploaded=st.file_uploader('Upload CSV file',type=['csv'])
    if uploaded is not None:
        try:
            raw=pd.read_csv(uploaded)
            st.caption(f'{len(raw)} records loaded')
            if len(raw)>5000: st.error('Batch maximum: 5,000 rows')
            elif st.button('Generate Batch Predictions',type='primary'):
                result=predict(raw)
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.download_button('⬇ Download Predictions',result.to_csv(index=False),file_name='credit_risk_batch_predictions.csv',mime='text/csv')
        except Exception as e: st.error(f'Batch assessment failed: {e}')

with tab3:
    st.markdown('### Global Feature Importance')
    st.info('This is overall XGBoost feature importance. It is not a local SHAP explanation, a causal finding or fairness certification.')
    try:
        transform=model.named_steps['preprocessing']
        classifier=model.named_steps['classifier']
        names=transform.get_feature_names_out()
        importance=pd.DataFrame({'Feature':names,'Importance':classifier.feature_importances_}).sort_values('Importance',ascending=False)
        top=importance.head(10).iloc[::-1]
        fig,ax=plt.subplots(figsize=(7.0,3.9))
        readable=top['Feature'].str.replace(r'^(numeric|categorical|num|cat)__','',regex=True)
        ax.barh(readable,top['Importance'],color=[BLUE if i%2==0 else RED for i in range(len(top))])
        ax.set_xlabel('XGBoost feature importance',fontweight='bold')
        ax.tick_params(axis='y',labelsize=7)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout()
        st.pyplot(fig,use_container_width=False);plt.close(fig)
        st.download_button('⬇ Download Feature Importance',importance.to_csv(index=False),file_name='credit_risk_feature_importance.csv',mime='text/csv')
    except Exception as e: st.error(f'Could not display model importance: {e}')

st.divider()
st.caption('Research prototype · German Credit (Statlog) · Frozen model · No automated lending decisions')
