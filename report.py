import pandas as pd
import numpy as np
import abc
import utils
from tqdm import tqdm
import config as cfg
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, calculate_statistics, calculate_linearization, ProportionZFromTest, UTestFromTest


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport:
    def __call__(self, calculated_metric, metric_items, mc_estimator = None) -> Report:

        mappings_estimator = {
            "t_test": TTestFromStats(),
            "mann_whitney": UTestFromTest(),
            "prop_test": ProportionZFromTest()
        }
        if mc_estimator:
            test = mappings_estimator[mc_estimator]
        else:
            test = mappings_estimator[metric_items.estimator]

        df_ = calculate_linearization(calculated_metric)
        stats = calculate_statistics(df_, metric_items.type)
        if metric_items.estimator == 'mann_whitney':
            criteria_res = test(df_)
        else: 
            criteria_res = test(stats)

        cfg.logger.info(f"{metric_items.name}")

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": stats.mean_1 - stats.mean_0,
            "lift":  (stats.mean_1 - stats.mean_0) / stats.mean_0,
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report(df, 
                            metric_config,
                            metric_montecarlo_config = None,
                            mc = None):
    build_metric_report = BuildMetricReport()
    reports = []
    if mc:
        for metric_params in metric_config:
            metric_parsed = Metric(metric_params, metric_montecarlo_config)
            lifts = metric_parsed.lift_list
            for lift in np.arange(lifts['start'], lifts['end'], step=lifts['by']):
                for estimator in  metric_parsed.mc_estimators_list:

                    cfg.logger.info(f"Metric: {metric_parsed.name} - Lift: {lift} - Estimator: {estimator}")
                    metric_reports = []

                    for _ in tqdm(range(1000)):
                        calculated_metric = CalculateMetric(metric_parsed)(df, lift)
                        metric_report = build_metric_report(calculated_metric, metric_parsed, mc_estimator=estimator)
                        metric_reports.append(metric_report.report)

                    metric_reports = pd.concat(metric_reports)
                    report_item = pd.DataFrame({
                        "metric_name": metric_parsed.name,
                        "lift": lift,
                        "estimator": estimator,
                        "TPR": sum(np.array(np.where(metric_reports['pvalue'] < 0.05)))/1000,
                        "agreement": sum(np.array(np.where(metric_reports['lift'] > 0)))/1000 
                    })
                    reports.append(Report(report_item).report)
    else:    
        for metric_params in metric_config:
            metric_parsed = Metric(metric_params)
            calculated_metric = CalculateMetric(metric_parsed)(df)
            metric_report = build_metric_report(calculated_metric, metric_parsed)
            reports.append(metric_report.report)

    return pd.concat(reports)





