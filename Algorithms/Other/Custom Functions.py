# ------------------------ Custom Function -------------------
# onehot encoding
def one_hot_encoder(data, categorical_features, nan_as_category=True):
    original_columns = list(data.columns)
    data = pd.get_dummies(data, columns=categorical_features, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    del original_columns
    return data, new_columns

# LightGBM with KFold to predict testset
# Stacking 第一层结构实现
# 二分类，AUC作为评判指标
def kfold_lightgbm(df, predictors, num_folds, stratified=False, debug=False, random_state=666,
                   learning_rate=0.1, n_estimators=2000,
                   num_leaves=31, colsample_bytree=0.7, subsample=0.7,
                   max_depth=-1, reg_alpha=0, reg_lambda=0, min_child_weight=1,
                   min_split_gain=0, max_bin=255, early_stopping_rounds=50):
    # Divide in training/validation and test data
    train_df = df[df['Set'] == 1]
    test_df = df[df['Set'] == 2]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=666)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=666)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    predictors = predictors

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[predictors], train_df['TARGET'])):
        train_x, train_y = train_df[predictors].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[predictors].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters
        clf = lgb.LGBMClassifier(nthread=4,  random_state=random_state, # is_unbalance=True,
                                 n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                 colsample_bytree=colsample_bytree, subsample=subsample,
                                 max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_split_gain=min_split_gain, min_child_weight=min_child_weight, silent=-1,
                                 verbose=2, max_bin=max_bin
                                 )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc', verbose=True,
                early_stopping_rounds=early_stopping_rounds
                # , scale_pos_weight=10
                )

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[predictors], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits # K折交叉得到sub_preds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = predictors
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # 输出整体AUC
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        submission = test_df[['SK_ID_CURR', 'TARGET']]
    # display_importances(feature_importance_df)

    return submission, feature_importance_df

# 自定义函数 F1-score-macro for lightgbm
def F1_Score_macro_froLGBM(y_true, y_pred):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = y_pred.reshape(len(np.unique(y_true)),-1).argmax(axis=0)
    f1 = f1_score(y_true=y_true, y_pred=pred_labels, average='macro')
    return ('F1_macro', f1, True)

# LightGBM with KFold to predict testset
# Stacking 第一层结构实现
# 多分类，F1作为评判指标
def kfold_lightgbm(train_x, 
                   train_y,
                   test, # test X
                   test_head, # test set id
                   num_folds, 
                   objective='multiclass', # ‘binary’ or ‘multiclass’ 
                   n_estimators=2000,
                   stratified=False, 
                   debug=False, 
                   random_state=666,
                   learning_rate=0.1, 
                   num_leaves=31, 
                   colsample_bytree=0.7, 
                   subsample=0.7,
                   max_depth=-1, 
                   reg_alpha=0, 
                   reg_lambda=0, 
                   min_child_weight=1,
                   min_split_gain=0, 
                   max_bin=255,
                   early_stopping_rounds=100
                  ):
    
    # Divide in training/validation and test data
    train_df_x = train_x
    train_df_y = train_y
    test_df = test
    print("Starting LightGBM. Train_x shape: {}, test shape: {}".format(train_df_x.shape, test_df.shape))
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=666)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=666)
    
    # Create arrays and dataframes to store results
    oof_preds = pd.DataFrame(np.zeros(shape=(train_df_x.shape[0], len(train_df_y.unique()))), columns=np.arange(0, len(train_df_y.unique())))  # 验证集（袋外）预测值，分折预测整合
    sub_preds = np.zeros((test_df.shape[0], len(train_df_y.unique())))   # 测试集预测值，全集去测取平均
    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df_x, train_df_y)):
        train_x, train_y = train_df_x.iloc[train_idx], train_df_y.iloc[train_idx]
        valid_x, valid_y = train_df_x.iloc[valid_idx], train_df_y.iloc[valid_idx]
        print('n_fold number : ', n_fold)
        
#         start = time.clock()

        # LightGBM parameters
        clf = lgb.LGBMClassifier(nthread=4,  random_state=random_state, 
                                 objective=objective, # is_unbalance=True,
                                 n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                 colsample_bytree=colsample_bytree, subsample=subsample,
                                 max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
                                 min_split_gain=min_split_gain, 
                                 min_child_weight=min_child_weight, 
                                 silent=-1,
                                 verbose=2, max_bin=max_bin,
                                 metric='None'
                                 )
        
        clf.fit(train_x, train_y, 
                eval_set=[(valid_x, valid_y)],  # [(valid_x, valid_y), (train_x, train_y)]
                eval_names=['Val'],   # ['Val', Train]
                eval_metric=F1_Score_macro_froLGBM, 
                verbose=10,
                early_stopping_rounds=early_stopping_rounds
                )
        
        
        df_valid = pd.DataFrame(clf.predict_proba(valid_x, num_iteration=clf.best_iteration_), index = valid_idx) # 验证集（袋外）预测值，分折预测整合
        oof_preds.iloc[valid_idx] = df_valid
        sub_preds += clf.predict_proba(test_df, num_iteration=clf.best_iteration_)/len(train_y.unique())    # K折交叉得到sub_preds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train_x.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d F1-macro : %.6f' % (n_fold + 1, f1_score(y_true=valid_y, y_pred=oof_preds.iloc[valid_idx].idxmax(axis=1), average='macro')))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
#         end = time.clock()
#         print(end-start)

    # 输出整体AUC
    print('Full F1-macro score %.6f' % f1_score(y_true=train_df_y, y_pred=oof_preds.idxmax(axis=1), average='macro'))

    # Write submission file and plot feature importance
    if not debug:
        sub_preds = pd.DataFrame(sub_preds).reset_index(drop=True)
        test_head = test_head.reset_index(drop=True)
        submission = pd.concat([test_head, sub_preds], axis=1)
        
    # display_importances(feature_importance_df)

    return submission, feature_importance_df

# Display/plot feature importance (to be fixed)
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

# 内存查看
info = psutil.virtual_memory()
print('内存使用：',psutil.Process(os.getpid()).memory_info().rss)
print('总内存：',info.total)
print('内存占比：',info.percent)
print('cpu个数：',psutil.cpu_count())