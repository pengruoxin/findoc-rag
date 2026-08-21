"""Build the sealed hard-v3 external question bank and annotation review packet.

This script freezes questions before any Agent run.  It deliberately does not
author reference answers: gold values and evidence pages require a subsequent,
independent annotation pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf

TaskType = Literal["extract", "compare", "calculate"]
ExpectedBehavior = Literal["answer", "abstain"]


@dataclass(frozen=True)
class QuestionSpec:
    slug: str
    task_type: TaskType
    agent_command: str
    query: str
    report_years: tuple[int, ...]
    challenge_types: tuple[str, ...]
    probes: tuple[str, ...]
    expected_behavior: ExpectedBehavior = "answer"
    expected_response_mode: str = "answer"


@dataclass(frozen=True)
class CompanyProfile:
    security_code: str
    company_name: str
    sector_hint: str
    specs: tuple[QuestionSpec, ...]
    future_query: str
    ambiguous_query: str


def q(
    slug: str,
    task_type: TaskType,
    command: str,
    query: str,
    years: tuple[int, ...],
    challenges: tuple[str, ...],
    probes: tuple[str, ...],
) -> QuestionSpec:
    return QuestionSpec(
        slug=slug,
        task_type=task_type,
        agent_command=command,
        query=query,
        report_years=years,
        challenge_types=challenges,
        probes=probes,
    )


EXACT_TABLE = ("document_blind", "dense_table", "exact_value", "multi_fact")
NARRATIVE = (
    "document_blind",
    "narrative_evidence",
    "cross_page_evidence",
    "multi_fact",
)
TREND = ("document_blind", "multi_document", "multi_year", "exact_value")
CALC = (
    "document_blind",
    "multi_document",
    "derived_metric",
    "calculator_required",
)
VERIFY = (
    "document_blind",
    "claim_verification",
    "multi_document",
    "multi_evidence",
)


PROFILES: tuple[CompanyProfile, ...] = (
    CompanyProfile(
        security_code="601398",
        company_name="工商银行",
        sector_hint="银行",
        specs=(
            q(
                "y23_core",
                "extract",
                "extract",
                "工商银行2023年净利润、资产总额、客户贷款及垫款总额和客户存款分别是多少？",
                (2023,),
                EXACT_TABLE + ("visual_summary",),
                ("净利润", "资产总额", "客户贷款及垫款总额", "客户存款"),
            ),
            q(
                "y23_asset_quality",
                "extract",
                "extract",
                "工商银行2023年不良贷款率、拨备覆盖率、资本充足率和成本收入比分别是多少？",
                (2023,),
                EXACT_TABLE + ("visual_summary",),
                ("不良贷款率", "拨备覆盖率", "资本充足率", "成本收入比"),
            ),
            q(
                "y23_ecl_audit",
                "extract",
                "extract",
                "工商银行2023年关于客户贷款及垫款预期信用损失的关键审计事项，风险点和审计应对分别是什么？",
                (2023,),
                NARRATIVE,
                ("关键审计事项", "预期信用损失", "客户贷款及垫款"),
            ),
            q(
                "y24_core",
                "extract",
                "extract",
                "工商银行2024年净利润、资产总额、客户贷款及垫款总额和客户存款分别是多少？",
                (2024,),
                EXACT_TABLE + ("visual_summary",),
                ("净利润", "资产总额", "客户贷款及垫款总额", "客户存款"),
            ),
            q(
                "y24_interest",
                "extract",
                "extract",
                "工商银行2024年净利息收入和净利息收益率如何变化，年报解释的主要原因是什么？",
                (2024,),
                NARRATIVE,
                ("净利息收入", "净利息收益率", "主要原因"),
            ),
            q(
                "trend_core",
                "compare",
                "compare",
                "比较工商银行2023年和2024年的净利润、资产总额、客户贷款及垫款总额和客户存款，分别判断增减。",
                (2023, 2024),
                TREND,
                ("净利润", "资产总额", "客户贷款及垫款总额", "客户存款"),
            ),
            q(
                "trend_asset_quality",
                "compare",
                "compare",
                "比较工商银行2023年和2024年的不良贷款率、拨备覆盖率、资本充足率和成本收入比。",
                (2023, 2024),
                TREND + ("visual_summary",),
                ("不良贷款率", "拨备覆盖率", "资本充足率", "成本收入比"),
            ),
            q(
                "calc_profit_growth",
                "calculate",
                "calculate",
                "用年报披露的净利润计算工商银行2024年相对2023年的增长率，保留两位小数。",
                (2023, 2024),
                CALC,
                ("净利润",),
            ),
            q(
                "calc_loan_deposit_gap",
                "calculate",
                "calculate",
                "分别计算工商银行2023年和2024年客户存款减客户贷款及垫款的差额，并判断差额是否扩大。",
                (2023, 2024),
                CALC,
                ("客户贷款及垫款总额", "客户存款"),
            ),
            q(
                "verify_quality",
                "compare",
                "verify",
                "核验说法：工商银行2024年较2023年不良贷款率下降，同时资本充足率上升。请给出证据和结论。",
                (2023, 2024),
                VERIFY + ("visual_summary",),
                ("不良贷款率", "资本充足率"),
            ),
        ),
        future_query="请给出工商银行2025年实际不良贷款率、拨备覆盖率和资本充足率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家银行两年的资产质量指标并解释变化，但先不要假定我指的是哪家银行或哪两个年度。",
    ),
    CompanyProfile(
        security_code="002594",
        company_name="比亚迪",
        sector_hint="新能源汽车",
        specs=(
            q(
                "y23_core",
                "extract",
                "extract",
                "比亚迪2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,),
                EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_products",
                "extract",
                "extract",
                "比亚迪2023年汽车、汽车相关产品及其他产品与手机部件、组装及其他产品的营业收入和毛利率分别是多少？",
                (2023,),
                EXACT_TABLE + ("cross_page_table",),
                ("汽车、汽车相关产品及其他产品", "手机部件、组装及其他产品", "毛利率"),
            ),
            q(
                "y23_rd",
                "extract",
                "extract",
                "比亚迪2023年研发投入金额、占营业收入比例、研发人员数量及占比分别是多少？",
                (2023,),
                EXACT_TABLE,
                ("研发投入金额", "研发投入占营业收入比例", "研发人员数量"),
            ),
            q(
                "y24_core",
                "extract",
                "extract",
                "比亚迪2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2024,),
                EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y24_products",
                "extract",
                "extract",
                "比亚迪2024年两大产品类别的营业收入、同比变化和毛利率分别是多少？",
                (2024,),
                EXACT_TABLE + ("cross_page_table",),
                ("分产品", "营业收入", "毛利率"),
            ),
            q(
                "trend_core",
                "compare",
                "compare",
                "比较比亚迪2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024),
                TREND,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_rd",
                "compare",
                "compare",
                "比较比亚迪2023年和2024年的研发投入金额、研发投入强度及研发人员数量。",
                (2023, 2024),
                TREND,
                ("研发投入金额", "研发投入占营业收入比例", "研发人员数量"),
            ),
            q(
                "calc_profit_margin",
                "calculate",
                "calculate",
                "用营业收入和归母净利润计算比亚迪2023年、2024年的归母净利率，并计算变化百分点。",
                (2023, 2024),
                CALC,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
            q(
                "calc_product_share",
                "calculate",
                "calculate",
                "计算比亚迪2024年汽车、汽车相关产品及其他产品收入占营业收入的比例，保留两位小数。",
                (2024,),
                CALC + ("dense_table",),
                ("汽车、汽车相关产品及其他产品", "营业收入"),
            ),
            q(
                "verify_growth",
                "compare",
                "verify",
                "核验说法：比亚迪2024年营业收入和归母净利润均高于2023年，且净利润增速更快。",
                (2023, 2024),
                VERIFY,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
        ),
        future_query="请给出比亚迪2025年全年汽车业务收入和毛利率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家新能源汽车公司的盈利能力变化，但先确认公司名称、比较年度和所指利润口径。",
    ),
    CompanyProfile(
        security_code="601318",
        company_name="中国平安",
        sector_hint="保险",
        specs=(
            q(
                "y23_core",
                "extract",
                "extract",
                "中国平安2023年归母营运利润、归母净利润，以及按2023年末最新投资回报率和风险贴现率等假设计算的寿险及健康险新业务价值和内含价值分别是多少？",
                (2023,),
                EXACT_TABLE,
                ("归属于母公司股东的营运利润", "归属于母公司股东的净利润", "新业务价值", "内含价值"),
            ),
            q(
                "y23_segments",
                "extract",
                "extract",
                "中国平安2023年寿险及健康险、财产保险、银行三项业务归属于母公司股东的营运利润分别是多少？",
                (2023,),
                EXACT_TABLE + ("segment_table",),
                ("寿险及健康险业务", "财产保险业务", "银行业务", "营运利润"),
            ),
            q(
                "y23_audit",
                "extract",
                "extract",
                "中国平安2023年保险合同负债计量相关关键审计事项的主要风险和审计应对是什么？",
                (2023,),
                NARRATIVE,
                ("关键审计事项", "保险合同负债", "审计应对"),
            ),
            q(
                "y24_core",
                "extract",
                "extract",
                "中国平安2024年归母营运利润、归母净利润，以及按2024年末最新假设计算的寿险及健康险新业务价值和内含价值分别是多少？",
                (2024,),
                EXACT_TABLE,
                ("归属于母公司股东的营运利润", "归属于母公司股东的净利润", "新业务价值", "内含价值"),
            ),
            q(
                "y24_customer",
                "extract",
                "extract",
                "中国平安2024年个人客户数、客均合同数以及持有四个及以上合同的客户留存率分别是多少？",
                (2024,),
                EXACT_TABLE + ("visual_summary",),
                ("个人客户", "客均合同", "留存率"),
            ),
            q(
                "trend_core",
                "compare",
                "compare",
                "按中国平安2024年年报的可比口径（其中2023年归母营运利润为追溯调整后），比较2023年和2024年的归母营运利润、归母净利润及寿险新业务价值。",
                (2023, 2024),
                TREND,
                ("归属于母公司股东的营运利润", "归属于母公司股东的净利润", "新业务价值"),
            ),
            q(
                "trend_customer",
                "compare",
                "compare",
                "比较中国平安2023年和2024年的个人客户数、客均合同数以及持有集团内4个及以上合同客户的留存率。",
                (2023, 2024),
                TREND,
                ("个人客户", "客均合同", "客户黏性"),
            ),
            q(
                "calc_nbv_growth",
                "calculate",
                "calculate",
                "按中国平安2024年年报的可比口径，用披露值计算2024年寿险及健康险新业务价值相对2023年的增长率。",
                (2023, 2024),
                CALC,
                ("新业务价值",),
            ),
            q(
                "calc_segment_share",
                "calculate",
                "calculate",
                "计算中国平安2024年寿险及健康险业务归母营运利润占寿险及健康险、财产保险、银行三项核心业务归母营运利润合计的比例。",
                (2024,),
                CALC + ("segment_table",),
                ("寿险及健康险业务", "财产保险业务", "银行业务", "营运利润"),
            ),
            q(
                "verify_nbv",
                "compare",
                "verify",
                "按中国平安2024年年报的可比（调整后）口径核验说法：2024年寿险及健康险新业务价值较2023年增长，但内含价值没有增长。",
                (2023, 2024),
                VERIFY,
                ("新业务价值", "内含价值"),
            ),
        ),
        future_query="请给出中国平安2025年寿险及健康险新业务价值和归母营运利润，并引用当前语料中的年报页码。",
        ambiguous_query="请分析这家保险公司两年的新业务价值变化，但先确认公司、年度以及新业务价值口径。",
    ),
    CompanyProfile(
        security_code="300750",
        company_name="宁德时代",
        sector_hint="动力电池",
        specs=(
            q(
                "y23_core", "extract", "extract",
                "宁德时代2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_products", "extract", "extract",
                "宁德时代2023年动力电池系统、储能电池系统和电池材料及回收三类业务的收入和毛利率分别是多少？",
                (2023,), EXACT_TABLE + ("segment_table",),
                ("动力电池系统", "储能电池系统", "电池材料及回收", "毛利率"),
            ),
            q(
                "y23_rd", "extract", "extract",
                "宁德时代2023年研发费用、研发投入占营业收入比例和研发人员数量分别是多少？",
                (2023,), EXACT_TABLE,
                ("研发费用", "研发投入占营业收入比例", "研发人员数量"),
            ),
            q(
                "y24_core", "extract", "extract",
                "宁德时代2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2024,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y24_products", "extract", "extract",
                "宁德时代2024年动力电池系统、储能电池系统和电池材料及回收业务的收入、同比变化和毛利率分别是多少？",
                (2024,), EXACT_TABLE + ("segment_table",),
                ("动力电池系统", "储能电池系统", "电池材料及回收", "毛利率"),
            ),
            q(
                "trend_core", "compare", "compare",
                "比较宁德时代2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024), TREND,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_products", "compare", "compare",
                "比较宁德时代2023年和2024年动力电池系统与储能电池系统的收入和毛利率。",
                (2023, 2024), TREND + ("segment_table",),
                ("动力电池系统", "储能电池系统", "毛利率"),
            ),
            q(
                "calc_storage_share", "calculate", "calculate",
                "计算宁德时代2024年储能电池系统收入占公司营业收入的比例，保留两位小数。",
                (2024,), CALC + ("segment_table",),
                ("储能电池系统", "营业收入"),
            ),
            q(
                "calc_net_margin", "calculate", "calculate",
                "计算宁德时代2023年和2024年的归母净利率，并给出变化百分点。",
                (2023, 2024), CALC,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
            q(
                "verify_divergence", "compare", "verify",
                "核验说法：宁德时代2024年营业收入较2023年下降，但归母净利润上升。",
                (2023, 2024), VERIFY,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
        ),
        future_query="请给出宁德时代2025年动力电池系统收入和毛利率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家电池公司的业务结构变化，但先确认公司、年度和所指电池业务口径。",
    ),
    CompanyProfile(
        security_code="600309",
        company_name="万华化学",
        sector_hint="化工",
        specs=(
            q(
                "y23_core", "extract", "extract",
                "万华化学2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_segments", "extract", "extract",
                "万华化学2023年聚氨酯、石化和精细化学品及新材料三类业务的收入和毛利率分别是多少？",
                (2023,), EXACT_TABLE + ("segment_table",),
                ("聚氨酯", "石化", "精细化学品及新材料", "毛利率"),
            ),
            q(
                "y23_rd", "extract", "extract",
                "万华化学2023年研发投入合计、费用化金额、资本化金额及占营业收入比例分别是多少？",
                (2023,), EXACT_TABLE,
                ("研发投入合计", "费用化研发投入", "资本化研发投入", "研发投入总额占营业收入比例"),
            ),
            q(
                "y24_core", "extract", "extract",
                "万华化学2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2024,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y24_segments", "extract", "extract",
                "万华化学2024年三类主要业务的收入、成本和毛利率分别是多少？",
                (2024,), EXACT_TABLE + ("segment_table",),
                ("聚氨酯", "石化", "精细化学品及新材料", "毛利率"),
            ),
            q(
                "trend_core", "compare", "compare",
                "比较万华化学2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024), TREND,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_segments", "compare", "compare",
                "比较万华化学2023年和2024年聚氨酯、石化、精细化学品及新材料业务的毛利率。",
                (2023, 2024), TREND + ("segment_table",),
                ("聚氨酯", "石化", "精细化学品及新材料", "毛利率"),
            ),
            q(
                "calc_polyurethane_share", "calculate", "calculate",
                "计算万华化学2024年聚氨酯业务收入占营业收入的比例，保留两位小数。",
                (2024,), CALC + ("segment_table",),
                ("聚氨酯", "营业收入"),
            ),
            q(
                "calc_margin_spread", "calculate", "calculate",
                "计算万华化学2024年聚氨酯业务与石化业务毛利率的差值，并与2023年的差值比较。",
                (2023, 2024), CALC + ("segment_table",),
                ("聚氨酯", "石化", "毛利率"),
            ),
            q(
                "verify_divergence", "compare", "verify",
                "核验说法：万华化学2024年营业收入较2023年增长，但归母净利润下降。",
                (2023, 2024), VERIFY,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
        ),
        future_query="请给出万华化学2025年聚氨酯业务收入和毛利率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家化工公司的分部毛利率变化，但先确认公司、比较年度和分部口径。",
    ),
    CompanyProfile(
        security_code="002352",
        company_name="顺丰控股",
        sector_hint="物流",
        specs=(
            q(
                "y23_core", "extract", "extract",
                "顺丰控股2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_businesses", "extract", "extract",
                "顺丰控股2023年各主要物流分部的营业收入及同比变化分别是多少？",
                (2023,), EXACT_TABLE + ("segment_table", "cross_page_table"),
                ("时效快递", "经济快递", "快运", "冷运及医药"),
            ),
            q(
                "y23_audit", "extract", "extract",
                "顺丰控股2023年收入确认相关关键审计事项的主要风险和审计应对是什么？",
                (2023,), NARRATIVE,
                ("关键审计事项", "收入确认", "审计应对"),
            ),
            q(
                "y24_core", "extract", "extract",
                "顺丰控股2024年营业收入、归母净利润、息税折旧摊销前利润和净资产收益率分别是多少？",
                (2024,), EXACT_TABLE + ("visual_summary",),
                ("营业收入", "归母净利润", "息税折旧摊销前利润", "净资产收益率"),
            ),
            q(
                "y24_network", "extract", "extract",
                "顺丰控股2024年国际快递、货代及供应链业务覆盖国家及地区数、全球运营管理干支线货车数、运营全货机数、全球累计运营航空线路数、铁路普列线路数和海运线路数分别是多少？请保留信息图中的大于号下限口径。",
                (2024,), EXACT_TABLE + ("visual_summary", "cross_page_evidence"),
                ("国际快递、货代及供应链", "全球运营管理干支线货车", "运营全货机", "全球累计运营航线", "铁路普列线路", "海运线路"),
            ),
            q(
                "trend_core", "compare", "compare",
                "比较顺丰控股2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024), TREND,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_businesses", "compare", "compare",
                "比较顺丰控股2023年和2024年时效快递、经济快递、快运及冷运医药业务的收入变化。",
                (2023, 2024), TREND + ("segment_table",),
                ("时效快递", "经济快递", "快运", "冷运及医药"),
            ),
            q(
                "calc_profit_margin", "calculate", "calculate",
                "计算顺丰控股2023年和2024年的归母净利率，并给出变化百分点。",
                (2023, 2024), CALC,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
            q(
                "calc_vehicle_rail_ratio", "calculate", "calculate",
                "根据顺丰控股2024年运力信息图披露的下限，计算全球运营管理干支线货车数量与铁路普列线路数量之比的下限，保留两位小数并保留大于号。",
                (2024,), CALC + ("visual_summary",),
                ("全球运营管理干支线货车", "铁路普列线路"),
            ),
            q(
                "verify_summary", "compare", "verify",
                "核验说法：顺丰控股2024年营业收入上升、归母净利润上升，但总资产下降。",
                (2023, 2024), VERIFY + ("visual_summary",),
                ("营业收入", "归母净利润", "总资产"),
            ),
        ),
        future_query="请给出顺丰控股2025年全球运营全货机数量和全年营业收入，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家物流公司的网络能力变化，但先确认公司、年度和要比较的运输方式。",
    ),
    CompanyProfile(
        security_code="300760",
        company_name="迈瑞医疗",
        sector_hint="医疗器械",
        specs=(
            q(
                "y23_core", "extract", "extract",
                "迈瑞医疗2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_products", "extract", "extract",
                "迈瑞医疗2023年生命信息与支持、体外诊断和医学影像三类产品的收入和毛利率分别是多少？",
                (2023,), EXACT_TABLE + ("segment_table",),
                ("生命信息与支持", "体外诊断", "医学影像", "毛利率"),
            ),
            q(
                "y23_rd", "extract", "extract",
                "迈瑞医疗2023年研发投入金额、占营业收入比例和研发人员数量分别是多少？",
                (2023,), EXACT_TABLE,
                ("研发投入金额", "研发投入占营业收入比例", "研发人员数量"),
            ),
            q(
                "y24_core", "extract", "extract",
                "迈瑞医疗2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2024,), EXACT_TABLE,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y24_products", "extract", "extract",
                "迈瑞医疗2024年三大产品线的收入、同比变化和毛利率分别是多少？",
                (2024,), EXACT_TABLE + ("segment_table",),
                ("生命信息与支持", "体外诊断", "医学影像", "毛利率"),
            ),
            q(
                "trend_core", "compare", "compare",
                "比较迈瑞医疗2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024), TREND,
                ("营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_products", "compare", "compare",
                "按迈瑞医疗2024年年报追溯调整后的可比口径，比较2023年和2024年生命信息与支持、体外诊断、医学影像三大产品线的收入与毛利率。",
                (2023, 2024), TREND + ("segment_table",),
                ("生命信息与支持", "体外诊断", "医学影像", "毛利率"),
            ),
            q(
                "calc_ivd_share", "calculate", "calculate",
                "计算迈瑞医疗2024年体外诊断产品收入占营业收入的比例，保留两位小数。",
                (2024,), CALC + ("segment_table",),
                ("体外诊断", "营业收入"),
            ),
            q(
                "calc_margin_spread", "calculate", "calculate",
                "按迈瑞医疗2024年年报追溯调整后的可比口径，计算2024年三大产品线最高与最低毛利率之差，并与2023年的差值比较。",
                (2023, 2024), CALC + ("segment_table",),
                ("生命信息与支持", "体外诊断", "医学影像", "毛利率"),
            ),
            q(
                "verify_growth", "compare", "verify",
                "核验说法：迈瑞医疗2024年营业收入和归母净利润均高于2023年。",
                (2023, 2024), VERIFY,
                ("营业收入", "归属于上市公司股东的净利润"),
            ),
        ),
        future_query="请给出迈瑞医疗2025年体外诊断产品收入和毛利率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家医疗器械公司的产品结构变化，但先确认公司、年度和产品分类口径。",
    ),
    CompanyProfile(
        security_code="000063",
        company_name="中兴通讯",
        sector_hint="通信设备",
        specs=(
            q(
                "y23_core", "extract", "extract",
                "中兴通讯2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2023,), EXACT_TABLE,
                ("营业收入", "归属于上市公司普通股股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y23_markets", "extract", "extract",
                "中兴通讯2023年运营商网络、政企业务和消费者业务的营业收入及毛利率分别是多少？",
                (2023,), EXACT_TABLE + ("segment_table",),
                ("运营商网络", "政企业务", "消费者业务", "毛利率"),
            ),
            q(
                "y23_rd", "extract", "extract",
                "中兴通讯2023年研发投入金额、占营业收入比例和研发人员占比分别是多少？",
                (2023,), EXACT_TABLE,
                ("研发投入", "占营业收入比例", "研发人员"),
            ),
            q(
                "y24_core", "extract", "extract",
                "中兴通讯2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？",
                (2024,), EXACT_TABLE,
                ("营业收入", "归属于上市公司普通股股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "y24_markets", "extract", "extract",
                "中兴通讯2024年三大业务市场的营业收入、同比变化和毛利率分别是多少？",
                (2024,), EXACT_TABLE + ("segment_table",),
                ("运营商网络", "政企业务", "消费者业务", "毛利率"),
            ),
            q(
                "trend_core", "compare", "compare",
                "比较中兴通讯2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。",
                (2023, 2024), TREND,
                ("营业收入", "归属于上市公司普通股股东的净利润", "经营活动产生的现金流量净额"),
            ),
            q(
                "trend_markets", "compare", "compare",
                "按中兴通讯2024年年报追溯调整后的可比口径，比较2023年和2024年运营商网络、政企业务和消费者业务的收入与毛利率。",
                (2023, 2024), TREND + ("segment_table",),
                ("运营商网络", "政企业务", "消费者业务", "毛利率"),
            ),
            q(
                "calc_enterprise_share", "calculate", "calculate",
                "计算中兴通讯2024年政企业务收入占营业收入的比例，保留两位小数。",
                (2024,), CALC + ("segment_table",),
                ("政企业务", "营业收入"),
            ),
            q(
                "calc_rd_change", "calculate", "calculate",
                "计算中兴通讯2024年研发投入相对2023年的增长率，并比较研发投入强度变化。",
                (2023, 2024), CALC,
                ("研发投入", "占营业收入比例"),
            ),
            q(
                "verify_divergence", "compare", "verify",
                "核验说法：中兴通讯2024年营业收入较2023年增长，但归母净利润下降。",
                (2023, 2024), VERIFY,
                ("营业收入", "归属于上市公司普通股股东的净利润"),
            ),
        ),
        future_query="请给出中兴通讯2025年政企业务收入和毛利率，并引用当前语料中的年报页码。",
        ambiguous_query="请比较这家通信设备公司的三大市场表现，但先确认公司、年度和市场分类口径。",
    ),
)


# Native extraction misses the labels embedded in this infographic.  The page
# was rendered and visually checked during question construction.
VISUAL_PAGE_OVERRIDES: dict[str, dict[str, list[int]]] = {
    "v3_002352_calc_vehicle_rail_ratio": {
        "cninfo:002352:annual:2024": [33],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path, default=Path("data/evaluation/agent-hard-v3-corpus-plan.json")
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-questions.json"),
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-question-review.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/evaluation/agent-hard-v3-question-bank-zh.md"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _question_hash(questions: list[dict]) -> str:
    blind_payload = [
        {
            "case_id": item["case_id"],
            "task_type": item["task_type"],
            "agent_command": item["agent_command"],
            "query": item["query"],
        }
        for item in questions
    ]
    encoded = json.dumps(
        blind_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_page_texts(source_records: dict[str, dict]) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    for key, record in source_records.items():
        with pymupdf.open(Path(record["local_file"])) as pdf:
            texts[key] = [page.get_text("text", sort=True) for page in pdf]
    return texts


def _candidate_pages(page_texts: list[str], probes: tuple[str, ...]) -> list[dict]:
    scored: list[tuple[int, int, list[str]]] = []
    for page_number, text in enumerate(page_texts, start=1):
        compact = "".join(text.split())
        matched = [probe for probe in probes if "".join(probe.split()) in compact]
        if matched:
            scored.append((len(matched), page_number, matched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "page_number": page_number,
            "matched_probes": matched,
            "probe_match_count": score,
        }
        for score, page_number, matched in scored[:6]
    ]


def build() -> tuple[dict, dict]:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    planned = {item["document_key"]: item for item in plan["documents"]}
    source_records = {
        item["document_key"]: item for item in manifest["documents"]
    }
    if set(source_records) != set(planned):
        raise ValueError("Source manifest does not exactly match the sealed corpus plan")

    profile_codes = {profile.security_code for profile in PROFILES}
    planned_codes = {item["security_code"] for item in plan["documents"]}
    if profile_codes != planned_codes:
        raise ValueError("Question profiles do not exactly match planned companies")

    page_texts = _load_page_texts(source_records)
    questions: list[dict] = []
    review_items: list[dict] = []
    split_by_code = {
        item["security_code"]: item["split"] for item in plan["documents"]
    }

    for profile in PROFILES:
        specs = list(profile.specs) + [
            QuestionSpec(
                slug="future_unavailable",
                task_type="extract",
                agent_command="extract",
                query=profile.future_query,
                report_years=(2025,),
                challenge_types=(
                    "document_blind",
                    "unavailable_period",
                    "abstention_required",
                ),
                probes=(),
                expected_behavior="abstain",
                expected_response_mode="insufficient_evidence",
            ),
            QuestionSpec(
                slug="ambiguous_scope",
                task_type="extract",
                agent_command="clarify",
                query=profile.ambiguous_query,
                report_years=(),
                challenge_types=(
                    "document_blind",
                    "ambiguous_scope",
                    "clarification_required",
                ),
                probes=(),
                expected_behavior="abstain",
                expected_response_mode="clarify",
            ),
        ]
        if len(specs) != 12:
            raise ValueError(f"{profile.company_name} must define exactly 12 questions")

        for spec in specs:
            case_id = f"v3_{profile.security_code}_{spec.slug}"
            questions.append(
                {
                    "case_id": case_id,
                    "split": split_by_code[profile.security_code],
                    "task_type": spec.task_type,
                    "agent_command": spec.agent_command,
                    "query": spec.query,
                    "challenge_types": list(spec.challenge_types),
                    "expected_behavior": spec.expected_behavior,
                    "expected_response_mode": spec.expected_response_mode,
                    "company_ids": [profile.security_code],
                    "company_name": profile.company_name,
                    "report_years": list(spec.report_years),
                    "gold_status": "pending_independent_annotation",
                }
            )

            source_keys = [
                f"cninfo:{profile.security_code}:annual:{year}"
                for year in spec.report_years
                if f"cninfo:{profile.security_code}:annual:{year}" in source_records
            ]
            evidence_candidates = []
            for source_key in source_keys:
                candidates = _candidate_pages(page_texts[source_key], spec.probes)
                visual_pages = VISUAL_PAGE_OVERRIDES.get(case_id, {}).get(source_key, [])
                candidates.extend(
                    {
                        "page_number": page_number,
                        "matched_probes": ["visual_manual_override"],
                        "probe_match_count": 0,
                    }
                    for page_number in visual_pages
                    if page_number not in {
                        candidate["page_number"] for candidate in candidates
                    }
                )
                evidence_candidates.append(
                    {
                        "document_key": source_key,
                        "local_file": source_records[source_key]["local_file"],
                        "candidate_pages": candidates,
                    }
                )
            structurally_verified = bool(source_keys) and all(
                item["candidate_pages"] for item in evidence_candidates
            )
            if spec.expected_behavior == "abstain":
                structurally_verified = True
            review_items.append(
                {
                    "case_id": case_id,
                    "query": spec.query,
                    "source_document_keys": source_keys,
                    "retrieval_probe_terms": list(spec.probes),
                    "evidence_candidates": evidence_candidates,
                    "structural_status": (
                        "visual_page_confirmed"
                        if case_id in VISUAL_PAGE_OVERRIDES
                        else "probe_pages_found"
                        if structurally_verified and spec.expected_behavior == "answer"
                        else "not_applicable"
                        if spec.expected_behavior == "abstain"
                        else "manual_source_check_required"
                    ),
                    "annotation_status": "gold_pending",
                    "review_checks": {
                        "query_is_answerable_as_written": None,
                        "source_page_confirmed": None,
                        "reference_facts_transcribed": None,
                        "calculation_recomputed": None,
                        "second_reviewer_approved": None,
                    },
                }
            )

    if len({item["case_id"] for item in questions}) != len(questions):
        raise ValueError("Question case IDs are not unique")
    if len({item["query"] for item in questions}) != len(questions):
        raise ValueError("Question text contains exact duplicates")

    command_counts = Counter(item["agent_command"] for item in questions)
    task_type_counts = Counter(item["task_type"] for item in questions)
    behavior_counts = Counter(item["expected_behavior"] for item in questions)
    split_counts = Counter(item["split"] for item in questions)
    challenge_counts = Counter(
        challenge for item in questions for challenge in item["challenge_types"]
    )
    question_hash = _question_hash(questions)
    source_hashes = sorted(item["sha256"] for item in manifest["documents"])
    source_set_hash = hashlib.sha256("\n".join(source_hashes).encode()).hexdigest()
    structural_status_counts = Counter(
        item["structural_status"] for item in review_items
    )

    question_bank = {
        "schema_version": "1",
        "dataset_id": "agent-hard-v3-external-questions",
        "status": "questions_frozen_gold_pending",
        "purpose": "Document- and company-blind external candidate set; no Agent run is permitted before gold freezing.",
        "corpus_plan": args.plan.as_posix(),
        "source_manifest": args.source_manifest.as_posix(),
        "question_payload_sha256": question_hash,
        "source_set_sha256": source_set_hash,
        "comparability": {
            "hard_v2_task_type_compatible": True,
            "same_three_core_task_types": True,
            "new_agent_commands": ["verify", "clarify"],
            "gold_scoring_ready": False,
            "reason_not_ready": "Independent gold facts, exact evidence pages, and double review are pending.",
        },
        "statistics": {
            "question_count": len(questions),
            "company_count": len(PROFILES),
            "source_document_count": len(source_records),
            "report_years": [2023, 2024],
            "task_type_counts": dict(sorted(task_type_counts.items())),
            "agent_command_counts": dict(sorted(command_counts.items())),
            "expected_behavior_counts": dict(sorted(behavior_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "challenge_type_counts": dict(sorted(challenge_counts.items())),
        },
        "questions": questions,
    }
    review_packet = {
        "schema_version": "1",
        "dataset_id": "agent-hard-v3-external-question-review",
        "question_payload_sha256": question_hash,
        "review_status": "question_structure_checked_gold_not_started",
        "independent_gold": False,
        "required_reviewers": 2,
        "instructions": [
            "Confirm the question against rendered source pages, not only native text.",
            "Transcribe reference facts without using DeepSeek or the evaluated Agent.",
            "For calculations, independently recompute inputs, formula, rounding, and unit.",
            "A second reviewer must verify question semantics, evidence, facts, and accepted variants.",
            "Do not expose this review packet or future gold to the Agent prompt.",
        ],
        "structural_status_counts": dict(sorted(structural_status_counts.items())),
        "items": review_items,
    }
    _write_json(args.questions, question_bank)
    _write_json(args.review_packet, review_packet)
    catalog_lines = [
        "# Agent hard-v3 外部候选题库",
        "",
        f"状态：`{question_bank['status']}`。当前只冻结题面，尚未制作独立 gold，也未运行 Agent。",
        "",
        f"题面 SHA-256：`{question_hash}`。",
        "",
        "题量：96；公司：8；官方年报：16（2023/2024）；可回答：80；拒答或澄清：16。",
        "",
        "任务分布：抽取 56、比较/核验 24、计算 16。以下清单不包含答案、证据页或目标文档 ID。",
        "",
    ]
    split_labels = {
        "calibration": "校准集",
        "dev": "开发集",
        "frozen_test": "冻结测试集",
    }
    for split in ("calibration", "dev", "frozen_test"):
        catalog_lines.extend([f"## {split_labels[split]}", ""])
        company_names = []
        for item in questions:
            if item["split"] == split and item["company_name"] not in company_names:
                company_names.append(item["company_name"])
        for company_name in company_names:
            catalog_lines.extend([f"### {company_name}", ""])
            for item in questions:
                if item["split"] != split or item["company_name"] != company_name:
                    continue
                catalog_lines.append(
                    f"- `{item['case_id']}` [{item['agent_command']}/{item['expected_response_mode']}] "
                    f"{item['query']}"
                )
            catalog_lines.append("")
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text("\n".join(catalog_lines).rstrip() + "\n", encoding="utf-8")
    return question_bank, review_packet


def main() -> None:
    question_bank, review_packet = build()
    print(f"questions={question_bank['statistics']['question_count']}")
    print(f"question_sha256={question_bank['question_payload_sha256']}")
    print(f"task_types={question_bank['statistics']['task_type_counts']}")
    print(f"commands={question_bank['statistics']['agent_command_counts']}")
    print(f"behaviors={question_bank['statistics']['expected_behavior_counts']}")
    print(f"structure={review_packet['structural_status_counts']}")


if __name__ == "__main__":
    main()
