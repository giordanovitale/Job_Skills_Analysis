import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


# ==================================================
# DATA EXPLORATION
# This section contains functions for data exploration.
# ==================================================


def get_cat_distribution(s, report_pct=True):
    """
    Gets distribution of a categorical series.

    Args:
        s (series): a series to summarize.
        report_pct (bool): return percentage if True. Otherwise, return count.

    Returns:
        Value counts for s.

    Authors:
        Tue Nguyen
    """
    return s.value_counts(dropna=False, normalize=report_pct).round(2) * 100


def plot_cat_distribution(s, top_n=10, report_pct=True, sort_index=False):
    """
    Plots the distribution of a categorical series.

    Args:
        s (series): a series to summarize.
        top_n (int): number of top categories to report.
        report_pct (bool): return percentage if True. Otherwise, return count.
        sort_index (bool): sort by index if True.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    # Get distribution
    dist = get_cat_distribution(s, report_pct).sort_values()

    # Top levels to show
    dist_to_show = dist.tail(top_n)

    if sort_index:
        dist_to_show = dist_to_show.sort_index()

    # Plot distribution
    figsize = 5, len(dist_to_show) * 0.3
    plt_title = f"Distribution of top {len(dist_to_show)} / {len(dist)}"

    if s.name:
        plt_title += f" - {s.name}"

    fig, ax = plt.subplots(figsize=figsize)
    dist_to_show.plot.barh(ax=ax)
    ax.set_title(plt_title)
    ax.set_xlabel("Percentage")
    ax.set_xlim(0, 100)
    plt.show()
    plt.close(fig)


def plot_num_distribution(s, showfliers=False):
    """
    Plots the distribution for a numerical series.

    Args:
        s (series): a numerical series.
        showfliers (bool): show outliers if True. Otherwise, hide outliers.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    # Config
    if showfliers:
        plt_title = "Box plot"
    else:
        plt_title = "Box plot (outliers removed)"

    fig, ax = plt.subplots(figsize=(5, 1))
    s.plot.box(ax=ax, vert=False, showfliers=showfliers)
    ax.set_title(plt_title)
    plt.show()
    plt.close(fig)


# ==================================================
# DATA CLEANING
# This section contains functions for data cleaning.
# ==================================================


def read_raw_data(filename, col_mapping, low_memory=True):
    """
    Reads a raw data file (CSV), subset columns, and rename columns.

    Args:
        filename (str): path to CSV file.
        col_mapping (dict): a dict contains old_col_name:new_col_name pairs
        low_memory (bool): infer col type based on sample data if True

    Returns:
        A data frame.

    Authors:
        Tue Nguyen
    """
    cols = col_mapping.keys()
    df = pd.read_csv(filename, low_memory=low_memory, usecols=cols)[cols]
    df = df.replace([None], np.nan).rename(columns=col_mapping)
    return df


def read_feather(filename):
    """
    Reads a feather file and and replace None with np.nan

    Args:
        filename (str): path to file.

    Returns:
        A data frame.

    Authors:
        Tue Nguyen
    """
    return pd.read_feather(filename).replace([None], np.nan)


def normalize_date(s):
    """
    Normalizes string representing dates in various formats into a unified
    format, i.e., 'yyyy-mm-dd'. If conversion is not possible, then returns
    the default value of '0001-01-01'.

    Args:
        s (str): a string representing a date.

    Returns:
        A string in the standard date format.

    Authors:
        Tue Nguyen
    """
    # Default return value if conversion is not possible
    full_date = "0001-01-01"

    # Standardize the date part separator
    s = re.sub("[/]", "-", str(s).strip())

    # Define possible date pattern
    patterns = [
        ("yyyy-mm-dd", "(\d{4})-((0?[1-9])|(1[0-2]))-((0?[1-9])|(1[0-2]))"),
        ("dd-mm-yyyy", "((0?[1-9])|(1[0-2]))-((0?[1-9])|(1[0-2]))-(\d{4})"),
        ("mm-yyyy", "((0?[1-9])|(1[0-2]))-(\d{4})"),
        ("yyyy-mm", "(\d{4})-((0?[1-9])|(1[0-2]))"),
        ("yyyy", "(\d{4})"),
    ]

    # Date conversion
    for d_format, d_pattern in patterns:
        d_pattern = re.compile(d_pattern)
        if re.match(d_pattern, s):
            full_date = get_date_by_format(s, d_format)
            break

    return full_date


def get_date_by_format(s, d_format):
    """
    Converts a string to the standardized form for a date based on the
    given format.

    Args:
        s (str): a string representing a date.
        d (str): current date format of s

    Returns:
        A string in the standard date format.

    Authors:
        Tue Nguyen
    """
    y, m, d = "0001", "01", "01"

    if d_format == "yyyy":
        y = s
    elif d_format == "mm-yyyy":
        m, y = s.split("-")
    elif d_format == "yyyy-mm":
        y, m = s.split("-")
    elif d_format == "dd-mm-yyyy":
        d, m, y = s.split("-")
    elif d_format == "yyyy-mm-dd":
        y, m, d = s.split("-")

    y = y.zfill(4)
    m = m.zfill(2)
    d = d.zfill(2)
    full_date = "-".join([y, m, d])

    return full_date


def replace_series(s, replacement_mapping):
    """
    Replaces elements in a series according to a replacement mapping.

    Args:
        s (series): a series of strings.
        replacement_mapping (dict): a mapping of replace_by:to_replace pairs.

    Returns:
        A series after replacement.

    Authors:
        Tue Nguyen
    """
    result = s.copy()
    for replace_by, to_replace in replacement_mapping.items():
        result.replace(to_replace, replace_by, regex=False, inplace=True)

    return result


def replace_if_startswith(x, pattern, replace_by):
    """Replace x by replace_by if x starts with pattern.

    Args:
        x (str): a string to check
        pattern (str): pattern
        replace_by (str): replacement

    Return:
        A string
    """
    if not isinstance(x, str):
        return x
    if x.startswith(pattern):
        return replace_by
    return x


def squeeze_series(s, sep=";"):
    """
    Squeezes a series by index and combine elements.

    Args:
        s (series): a series to squeeze.
        sep (str): a string for the separator.

    Returns:
        A series after squeezed.

    Authors:
        Tue Nguyen
    """
    return s.reset_index().groupby("index").agg(set).squeeze().str.join(";")


def normalize_city_series(col, sep=None):
    """
    Normalizes city column.

    Args:
        col (series): a series of string (locations) to normalize.
        sep (str): regex string for location separators.

    Returns:
        A series of standardized cities.

    Authors:
        Tue Nguyen
    """
    # Default value for location separators if not provided
    if not sep:
        sep = "[-,;(/[]"

    # Replacement mapping in the format replace_by:to_replace
    replacement_mapping = {
        "BARCELONA": ["BARCELLONA"],
        "DUBLIN": ["DUBLINO"],
        "EUROPE": ["EU", "EUROPEROPE"],
        "FLORENCE": ["FIRENZE"],
        "GENOA": ["GENOVA"],
        "GENEVE": ["GINEVRA"],
        "ITALY": ["ITALY MULTIPLE LOCATION"],
        "REMOTE": ["ANCHE REMOTO", "FULL REMOTE"],
        "LONDON": ["LONDAN", "LONDRA"],
        "LAUSSANE": ["LOSANNA"],
        "LUXEMBOURG": ["LUSSEMBURGO"],
        "MILAN": ["MI", "MILANO", "MILANO ROZZANO", "MILANO OR REMOTE"],
        "SWITZERLAND": ["CH"],
    }

    # Explode each cell separate locations and standardize them
    locations = (
        col.str.split(sep)
        .explode()
        .str.strip()
        .str.upper()
        .replace("[)\]?]", "", regex=True)
        .replace([""], np.nan, regex=False)
    )

    # Perform replacement and squeeze
    locations = replace_series(locations, replacement_mapping)
    locations = squeeze_series(locations)

    return locations


def normalize_country_series(col, sep=None):
    """
    Normalizes country column.

    Args:
        col (series): a series of string (locations) to normalize.
        sep (str): regex string for location separators.

    Returns:
        A series of standardized countries.

    Authors:
        Tue Nguyen
    """
    # Default value for location separators if not provided
    if not sep:
        sep = "[-,;(/[]"

    # Replacement mapping in the format replace_by:to_replace
    replacement_mapping = {
        "IT": [
            "ITALY",
            "ITALIA",
            "ITALT",
            "ITALY",
            "ITLAY",
            "`ITALY",
            "ITALIA O SVIZZERA",
            "IRALT",
            "MILAN",
            "ROMA",
            "VENETO",
        ],
        "AR": ["ARGENTINA"],
        "AU": ["AUSTRALIA"],
        "BE": ["BELGIO"],
        "FR": ["FRANCE"],
        "DE": ["GERMANY"],
        "HU": ["HUNGARY"],
        "IE": ["IRELAND"],
        "IL": ["ISRAEL"],
        "JP": ["JO"],
        "GB": ["LONDON", "SCOZIA", "UK", "UNITED KINGDOM"],
        "US": ["USA"],
        "LU": ["LUXEMBOURG"],
        "MT": ["MALTA"],
        "MD": ["MOLDOVA"],
        "NO": ["NORWAY"],
        "PL": ["POLAND"],
        "RS": ["SERBIA"],
        "SK": ["SLOVACCHIA"],
        "ES": ["SPAIN"],
        "CH": ["SVIZZERA", "SWITZERLAND"],
        "SE": ["SWEDEN"],
        "TR": ["TURKEY"],
        "UA": ["UKRAINE"],
    }

    # Explode each cell separate locations and standardize them
    locations = (
        col.str.split(sep)
        .explode()
        .str.strip()
        .str.upper()
        .replace([""], np.nan, regex=False)
    )

    # Perform replacement
    locations = replace_series(locations, replacement_mapping)
    locations = squeeze_series(locations)

    return locations


def normalize_str_separated_series(s, sep=None):
    """
    Normalize a string series with token separated by sep.

    Args:
        s (series): a series to normalize.
        sep (str): separator.

    Returns:
        A normalized series.

    Authors:
        Tue Nguyen
    """
    if not sep:
        sep = "[,;]"

    s = s.str.split(sep).explode().str.strip().str.upper()
    return squeeze_series(s)


def normalize_skills_series(col, sep=None):
    """
    Normalizes skills column.

    Args:
        col (series): a series of strings (skills) to normalize.
        sep (str): regex string for skill separators.

    Returns:
        A series of standardized skills.

    Authors:
        Tue Nguyen
    """
    # Default value for location separators if not provided
    if not sep:
        sep = "[;,(/&]"

    # Replacement mapping in the format replace_by:to_replace
    replacement_mapping = {
        "BIG DATA": ["BIGDATA"],
        "SPARK": ["APACHE SPARK", "SPARQL"],
        "ARTIFICIAL INTELLIGENCE": ["AI"],
        "AZURE": ["AZURE AD", "AZURE PLATFORM"],
        "BASH": ["BASH SCRIPTING"],
        "DATA SCIENCE": ["DATA SCIENTIST"],
        "DATA VISUALIZATION": ["DATA VISUALISATION"],
        "BUSINESS ANALYSIS": ["BUSINESS ANALYST"],
        "BUSINESS INTELLIGENCE": ["BI MICROSOFT", "BI"],
        "CONFIGURATION": ["CONFIGURATION AND MONITORING"],
        "SW DEVELOPMENT": [
            "DEVELOPERS",
            "DEVELOPER",
            "ENGINEERING AND SOFTWARE ARCHITECTURE DESIGN",
        ],
        "DIGITAL INTELLIGENCE": ["DI"],
        "FINTECH": ["FINTECH ANALYSIS"],
        "HTML": ["HTML5"],
        "JSON": ["JSON LIBRARIES"],
        "LEAD": [
            "LEADERSHIP AND COLLABORATION",
            "TEAM LEADER",
            "TEAM MANAGEMENT",
            "TECHNICAL LEAD",
        ],
        "MARKETING": ["MARKETO"],
        "DATA": ["MASTER DATA", "DATA OFFICER", "DATA ROADMAP"],
        "DATA MANAGEMENT": ["MASTER DATA MANAGEMENT", "DATA MANAGER"],
        "PYTHON": ["PHYTON", "PYTHONI", "SKILLS:\xa0PYTHON"],
        "QLIK": ["QLIKVIEW"],
        "PREDICTIVE ANALYSIS": ["PREDICTIVE"],
        "NOSQL": ["REDIS AND OTHER NOSQL DATA BASE"],
        "PROJECT MANAGEMENT": ["PROJECT MANAGER", "PROJECT MANAGERS"],
        "SQL": [
            "MICROSOFT SQL SERVER",
            "MICROSOFT SQLSERVER",
            "DB2",
            "ORACLEDB",
            "POSTGRESQL",
            "MYSQL",
            "RDBMS",
            "DATABASE MYSQL",
            "DATABASE SQL",
            "POSTEGRESQL",
            "POSTGERSQL",
            "POSTGESQL",
        ],
        "MOBILE": ["MOBILE APPS", "MOBILE FRAMEWORKS"],
        "NLP": ["NATURAL LANGUAGE UNDERSTANDING", "NATURAL LANGUAGE PROCESSING"],
        "CLOUD": [
            "CLOUD COMPUTING",
            "CLOUD DEVELOPMENT",
            "GOOGLE CLOUD",
            "GOOGLE CLOUD PLATFORM",
            "MINIMAL EXPERIENCE WITH CLOUD ADMIN",
        ],
        "FUNCTIONAL ANALYSIS": [
            "FUNCTIONAL ANALYST",
            "FUNCTIONAL ANALYSYS",
            "ANALISTA FUNZIONALE",
        ],
        "CONSULTANCY": [
            "CONSULENTE",
            "CONSULTANTS",
            "CONSULTANT",
            "CONSULTING",
            "CONSULTATIVE BUSINESS DEVELOPMENT",
            "SKILLS: CONSULTING",
            "SENIOR CONSULTANT",
        ],
        "RESOURCE DESCRIPTION FRAMEWORK": ["RDF"],
        "API;REST": ["API GATEWAY", "API REST", "REST"],
        "SAP": [
            "SAP BW",
            "SAP APO",
            "SAP BI",
            "SAP BO",
            "SAP BPC",
            "SAP CO",
            "SAP CRM",
            "SAP FI",
            "SAP HANA",
            "SAP MM",
            "SAP PP",
            "SAP SD",
            "SAP SYCLO",
            "4HANA",
        ],
        "ACCENTURE": ["ACCENTURE INTERACTIVE"],
        "AGILE": ["AGILE SOFTWARE DEVELOPMENT"],
        "ANDROID": ["ANDROID DEVELOPMENT", "ANDRIOD", "ANDORID", "ANDR", "ANDRIOD SDK"],
        "ANGULAR JS": [
            "ANGULAR.JS",
            "ANGULAR2",
            "ANGULARJS",
            "ANGOLARJS",
            "ANGULA",
            "ANGULSRJS",
        ],
        "ASSET MANAGEMENT": ["AM"],
        "AWS;AZURE": ["AWS OR AZURE"],
        "BLOCKCHAIN": ["BLOCKCHAIN DEVELOPMENT"],
        "BLUE PRISM;AUTOMATION ANYWHERE": ["BLUE PRISM OR AUTOMATION ANYWHERE"],
        "CARD PAYMENTS;CASHLESS PAYMENTS": ["CARDS", "CASHLESS PAYMENTS"],
        "CHIEF DATA OFFICER": ["CDO"],
        "CSS": ["CSS3", "CASCADING STYLE SHEETS"],
        "D3 JS": ["D3", "D3JS"],
        "DATA ENGINEERING": ["DATA ENGINEER"],
        "DATA GOVERNANCE": ["DATA GOVERNANCE SOFTWARE"],
        "SECURITY": [
            "IT SECURITY",
            "DEFINE AND SUPPORT IT SECURITY POLICY",
            "KNOWLEDGE OF SECURITY SYSTEMS AND ARCHITECTURES",
            "SECURITY",
        ],
        "DISTRIBUTED COMPUTING": ["DISTRIBUTED ARCHITECTURES"],
        "DOCKER;KUBERNETES": ["DOCKER OR KUBERNETES"],
        "DATA WAREHOUSING": ["DWH"],
        "WEB": [
            "E UN WEB ARCHITECT",
            "WEBLOGIC 103",
            "STANNO CERCANDO 2 BACK END E 2 FULL STACK DEVELOPER",
        ],
        "CLOUD;AWS;E2C": ["EC2"],
        "BLOCKCHAIN;ETHEREUM": ["ETHEREUM"],
        "FINANCE;FINANCIAL ANALYSIS": ["FINANCIAL ANALYSIS"],
        "FINANCE;FINANCIAL FRAMEWORKS": ["FINANCIAL FRAMEWORKS"],
        "GIT;REACT": ["GIT AND A MINUMUM OF 2 YEARS EXPERIENCE WITH REACT"],
        "GIT": ["GITHUB"],
        "HADOOP": ["HDFS", "HADHOOP", "HADO"],
        "IT": ["INFORMATICA", "IT SUPPORT ON MAC AND MICROSOFT HARDWARE"],
        "INFRASTRUCTURE": [
            "INFRASTRUCTURE EXPERIENCE",
            "INFRASTRUCTURE INTEGRATION",
            "SYSTEM INTEGRATIONS",
        ],
        "ISO 27001;IT SECURITY": ["ISO 27001", "ISO27001LA", "IEC 27001"],
        "JAVA": ["JAVA EE", "RXJAVA", "SKILLS: JAVA", "J2EE"],
        "JAVASCRIPT": ["JS"],
        "MICROSOFT BI;BUSINESS INTELLIGENCE": ["MICROSOFT BI"],
        "POWER BI;BUSINESS INTELLIGENCE": ["POWER BI"],
        "MONGODB": ["MONGO"],
        "MICROSOFT AD": ["MS AD AUTHENTICATION"],
        "NEGOTIATION": ["CONTRACT AND VENDOR NEGOTIATIONS", "NEGOTIATIONS"],
        "SALES": ["PRESALES", "PRE SALES", "PRE-SALES AND BUDGET MANAGEMENT"],
        "PROJECT MANAGEMENT;PRODUCT MANAGEMENT": ["PROJECT AND PRODUCT MANAGEMENT"],
        "RECOMMENDATION": ["RECCOMENDATION"],
        "ROBOTIC": ["ROBOTIC PROCESS AUTOMATION"],
        "AWS S3;AWS": ["S3"],
        "SALESFORCE": ["SALESFORCECOM"],
        "SAS": ["SAS VISUAL ANALYTICS", "SASS"],
        "PYTHON;SCIKITLEARN": ["SCIKITLEARN"],
        "PYTHON;SCIPY": ["SCIPY"],
        "STATISTICS": ["STATISTICAL ANALYSIS"],
        "SWIFT": ["SWIFT 5", "SWIFT SECURITY PROGRAMME"],
        "VIRTUAL MACHINES": ["VIRTUAL MACHINES"],
        "SECURITY;VULNERABILITY ASSESSMENT": ["VULNERABILITY ASSESSMENT"],
        "SECURITY;VULNERABILITY MANAGEMENT": ["VULNERABILITY MANAGEMENT"],
        "VUE JS": ["VUE"],
        "WEBPACK": ["WEBPACK OR SIMILAR TOOLS"],
        "WEBSHEPERE": ["WEBSHEPERE 7"],
        "PIG;SPARK": ["PIG AND SPARK"],
        "ANALYSIS": ["ANALYSE", "ANALYSING"],
        "BOOTSTRAP": [
            "BOOTSRAP",
            "BOOTRSTRAPS",
            "BOOTRSTRAPS",
            "BOOSTRAPCSS",
            "BOOTSTR",
            "BOOTSTR",
            "BOOST",
        ],
        "CASSANDRA": ["CASANDRA"],
        "DATABASE": ["DBMS", "DBASE", "DBA"],
        "MICROSOFT": ["MICOROSOFT", "MICORSOFT", "MICORSOFT EXCEL"],
    }

    # Explode each cell separate locations and standardize them
    skills = (
        col.str.split(sep)
        .explode()
        .replace("[)\]?.]", "", regex=True)
        .replace('"', "", regex=True)
        .replace("'", "", regex=True)
        .str.strip()
        .str.upper()
        .replace([""], np.nan, regex=False)
    )

    # Perform replacement and squeeze
    skills = replace_series(skills, replacement_mapping)
    skills = skills.str.split(";").explode().str.strip()
    skills = squeeze_series(skills)

    return skills


def normalize_skills_series_extra(s, sep=None):
    """Normalize extra for candidate IT skills

    Args:
        s (series): a series
        sep (str): separator

    Returns:
        A series
    """
    # Normalize first steps
    s = normalize_skills_series(s, sep=sep)

    # Replacement mapping in the format replace_by:to_replace
    replacement_dict = {
        "2G NET": "2G NETWORKS",
        "3G NET": "3G NETWORKS",
        "4G NET": "4G NETWORKS",
        "5G NET": "5G NETWORKS",
        "6G NET": "6G NETWORKS",
        "3D STUDIO": "3D STUDIO",
        "ACCEPTANCE TEST": "ACCEPTANCE TEST",
        "ACTIONSCR": "ACTIONSCRIPT",
        "ACTIVE DI": "ACTIVE DIRECTORY",
        "ACTIVE S": "ACTIVE SERVER",
        "ACTIVE TEMPLATE": "ACTIVE TEMPLATE",
        "ACTIVEMATRI": "ACTIVEMATRIX ",
        "ACTIVITI": "ACTIVITI",
        "AD HOC": "ADHOC ANALYSIS",
        "ADHOC": "ADHOC ANALYSIS",
        "AD ": "ADVERTISEMENT",
        "ADA ": "ADA",
        "ADABAS ": "ADABAS",
        "ADOBE ": "ADOBE",
        "AGILE ": "AGILE",
        "AJAX ": "AJAX",
        "AKKA ": "AKKA",
        "ALGO": "ALGORITHMS",
        "ALL IT": "IT",
        "ALTOVA ": "ALTOVA",
        "AMAZO": "AWS",
        "ANDRO": "ANDROID",
        "ANGULA": "ANGULAR JS",
        "APACHE HADOOP": "HADOOP",
        "APACHE HBASE": "HBASE",
        "APACHE HI": "HIVE",
        "APACHE K": "KAFKA",
        "APACHE M": "MAHOUT",
        "APACHE SP": "SPARK",
        "APACHE SU": "APACHE SUBVERSION",
        "APACHE WE": "APACHE WEB SERVER",
        "API ": "API",
        "APOLLO ": "APOLLO",
        "APPLE": "APPLE",
        "ARTIFIC": "ARTIFICIAL INTELLIGENCE",
        "ASCENTIAL": "ASCENTIAL DATASTAGE",
        "ASPEN ": "ASPEN",
        "ASSET ": "ASSET MANAGEMENT",
        "ASYNCHR": "ASYNCHRONOUS TRANSFER MODE",
        "ATLASSIAN": "ATLASSIAN",
        "AUTODESK": "AUTODESK",
        "AWS ": "AWS",
        "AZURE ": "AZURE",
        "BACK-END": "BACK END",
        "BACKBONE NETWORKS": "BACKBONE NETWORKS",
        "BALSAMIC": "BALSAMIQ",
        "BASE SAS": "SAS",
        "BEX ": "BEX",
        "BIG D": "BIG DATA",
        "BLOCKCHA": "BLOCKCHAIN",
        "BLOOMBE": "BLOOMBERG",
        "BLUETOO": "BLUETOOTH",
        "BOOTSTRAP": "BOOTSTRAP",
        "CASCADING STYLE": "CSS",
        "ARCHITEC": "ARCHITECTURE",
        "ARDUI": "ARDUINO",
        "ARM ": "ARM",
        "ART ": "ART",
        "ASAP ": "ASAP",
        "ASP ": "ASP",
        "BASH ": "BASH",
        "BEHAVIOUR": "BEHAVIOUR ANALYSIS",
        "BILLIN": "BILLING SYSTEMS",
        "BITBUC": "BITBUCKET",
        "BLACKBERRY": "BLACKBERRY",
        "BMC ": "BMC",
        "BOOSTRAP": "BOOTSTRAP",
        "BOURNE": "BOURNE",
        "BUSINESS O": "BUSINESS OBJECT",
        "BUSINESSOB": "BUSINESS OBJECT",
        "BUSINESS S": "BUSINESS SUPPORT",
        "C ": "C",
        "C+": "C++",
        "CAM ": "CAM",
        "CASCA": "CSS",
        "CCS ": "CCS",
        "CASSANDRA": "CASSANDRA",
        "CD ": "CD",
        "CELLULAR ": "CELLULAR",
        "CENTOS ": "CENTOS",
        "CGI ": "CGI",
        "CISCO ": "CISCO",
        "CITRIX ": "CITRIX",
        "CLOUD ": "CLOUD",
        "CLOUDE": "CLOUDERA",
        "CLUSTER ": "DISTRIBUTED COMPUTING",
        "CLUSTERS": "DISTRIBUTED COMPUTING",
        "CMMI": "CMMI",
        "CMS ": "CMS",
        "COBOL": "COBOL",
        "COCOA": "COCOA",
        "CODEIG": "CODEIGNITER",
        "COGNOS": "COGNOS",
        "COMMUNICAT": "COMMUNICATION",
        "CONSUL": "CONSULTANCY",
        "CONTENT M": "CONTENT MANAGEMENT",
        "CONTINUOUS D": "CD",
        "CONTINUOUS I": "CI",
        "CRM ": "CRM",
        "CRUD ": "CRUD",
        "DATA I": "DATA INTEGRATION",
        "DATA MIN": "DATA MINING",
        "DATA MO": "DATA MODELLING",
        "DATA M": "DATA MANAGEMENT",
        "DATA PRO": "DATA PROCESSING",
        "DATA PRE-PROCESSING": "DATA PROCESSING",
        "DATA QUALITY": "DATA MANAGEMENT",
        "DATA STORAG": "DATA STORAGE",
        "DATA TRANS": "DATA TRANSFORMATION",
        "DATA VI": "DATA VISUALIZATION",
        "DATA WA": "DATA WAREHOUSING",
        "DATABASE ": "DATABASE",
        "DEB": "DEBIAN",
        "DEEP": "DEEP LEARNING",
        "DEL ": "DEL",
        "DEPLOYMENT ": "DEPLOYMENT",
        "DISTR": "DISTRIBUTED COMPUTING",
        "DJANGO": "DJANGO",
        "DOCKER": "DOCKER",
        "DOJO": "DOJO",
        "DOMAIN NA": "DNS",
        "DREAMWEAVER": "DREAMWEAVER",
        "DRUPAL": "DRUPAL",
        "E-COMMERCE": "E-COMMERCE",
        "ECOMME": "E-COMMERCE",
        "ECLIPSE": "ECLIPSE",
        "ELASTICSEAR": "ELASTICHSEARCH",
        "EMBED": "EMBEDDED",
        "ERP ": "ERP",
        "ETL ": "ETL",
        "EXTREME PR": "EXTREME PROGRAMMING",
        "FIREWA": "FIREWALLS",
        "FLASH ": "FLASH",
        "FRONT E": "FRONT END",
        "FRONTE": "FRONT END",
        "FTTX ": "FTTX",
        "FULL ST": "SW DEVELOPMENT",
        "FUNCTIONAL D": "FUNCTIONAL DESIGN",
        "FUNCTIONAL PRO": "FUNCTIONAL PROGRAMMING",
        "FUNCTIONAL RE": "FUNCTIONAL REQUIREMENTS",
        "GAME ": "GAME",
        "GEOGRAPHIC I": "GEOGRAPHIC INFORMATION SYSTEMS",
        "GOOGLE CL": "CLOUD",
        "GRAPHIC": "GRAPHICS",
        "GUI ": "GUI",
        "HADOOP ": "HADOOP",
        "HANA ": "HANA",
        "HELPDESK": "HELPDESK",
        "HIBE": "HIBERNATE",
        "HP ": "HP",
        "HTM": "HTML",
        "HTTP": "HTTP",
        "HYPERION": "HYPERION",
        "IBM CLOUD": "CLOUD",
        "IBM COGNO": "COGNOS",
        "IBM HA": "IBM HARDWARE",
        "IBM WEB": "IBM WEBSPHERE",
        "ICT ": "ICT",
        "IDEE": "IDEE",
        "INFORMATICA": "IT",
        "INFORMATION SEC": "SECURITY",
        "INFORMATION TEC": "IT",
        "INTEL ": "INTEL",
        "INTELLIJ": "INTELLIJ",
        "INTRUSION": "INTRUSION DETECTION",
        "IONIC ": "IONIC",
        "IOT ": "IOT",
        "IP ": "IP",
        "IPHONE": "IPHONE",
        "ISO 27": "SECURITY",
        "ISO 900X": "ISO 900X",
        "IT ARCHITECTURE": "IT ARCHITECTURE",
        "IT INFRASTRUCTURE": "INFRASTRUCTURE",
        "IT PROJECT": "PROJECT MANAGEMENT",
        "ITIL": "ITIL",
        "JAK": "JAKARTA TOMCAT",
        "JASPER": "JASPER",
        "JAVA ": "JAVA",
        "JAVASC": "JAVASCRIPT",
        "JAVASE": "JAVA SERVER",
        "JBOS": "JBOSS",
        "JDE": "JDE",
        "JIRA ": "JIRA",
        "JOOM": "JOOMLA",
        "JQU": "JQUERY",
        "JSC": "JAVASCRIPT",
        "JSO": "JSON",
        "JULIA": "JULIA",
        "JUPITER": "JUPYTER",
        "JUPYTER": "JUPYTER",
        "KAFK": "KAFKA",
        "KERA": "KERAS",
        "KNOC": "KNOCKOUT JS",
        "KORN ": "KORN",
        "LINU": "LINUX",
        "LOTUS ": "LOTUS",
        "LSI ": "LSI",
        "LUA": "LUA",
        "MACHINE L": "MACHINE LEARNING",
        "MACROMEDIA": "MACROMEDIA",
        "MARKET": "MARKETING ",
        "MASTER DATA": "DATA",
        "MATLA": "MATLAB",
        "MCAFEE": "MCAFEE",
        "MERCURY ": "MERCURY",
        "MICROSOFT AS": "ASP",
        "MICROSOFT C": "C#",
        "MICROSOFT DY": "MICROSOFT DYNAMICS",
        "MICROSOFT E": "MS EXCEL",
        "MS EXCEL": "MS EXCEL",
        "MICROSOFT OF": "MS OFFICE",
        "MICROSOFT OU": "MS OUTLOOK",
        "MICROSOFT PO": "MS POWERPOINT",
        "MICROSOFT SQ": "SQL",
        "MICROSOFT TRA": "SQL",
        "MICROSOFT TEAM": "MS TEAMS",
        "MICROSOFT VISI": "VISIO",
        "MICROSOFT VISUAL C": "VISUAL C",
        "MICROSOFT VISUAL S": "VISUAL STUDIO",
        "MICROSOFT WI": "WINDOWS",
        "MICROSOFT WO": "MS WORD",
        "MOBILE ": "MOBILE",
        "MVC ": "MVC",
        "MYSQ": "SQL",
        "NATURAL LA": "NLP",
        "NETBEAN": "NETBEANS",
        "NEURAL NE": "NEURAL NET",
        "NODE": "NODE JS",
        "OBJECT ORI": "OOP",
        "OPERATI": "OS",
        "ORACLE ": "ORACLE",
        "PHP ": "PHP",
        "POSTGR": "SQL",
        "PROJECT MANAGEMENT": "PROJECT MANAGEMENT",
        "PYTH": "PYTHON",
        "PYTOR": "PYTORCH",
        "QLIK": "QLIK",
        "R ": "R",
        "REACT": "REACT JS",
        "RECOVERY": "RECOVERY",
        "REGRESSION": "REGRESSION",
        "REGULAR EXP": "REGULAR EXPRESSION",
        "RELATIONAL": "SQL",
        "REST ": "API",
        "RUBY": "RUBY",
        "SAGE ": "SAGE",
        "SALESFORCE": "SALESFORCE",
        "SALES ": "SALES",
        "SAP ": "SAP",
        "SAS ": "SAS",
        "SCALA ": "SCALA",
        "SCIKI": "SCIKITLEARN",
        "SCO ": "SCO",
        "SCRUM": "SCRUM",
        "SEAGATE ": "SEAGATE",
        "SEARCH ENG": "SEO",
        "SECURITY": "SECURITY",
        "SERVER ": "SERVER ",
        "SHELL ": "SHELL",
        "SIEBEL ": "SIEBEL",
        "SIEMEN": "SIEMENS",
        "SNOWFLAKE": "SNOWFLAKE",
        "SOA ": "SOA",
        "SOAP ": "SOAP",
        "SOCIAL MED": "SOCIAL MEDIA",
        "SOFTWARE ": "SW DEVELOPMENT",
        "SPRING ": "SPRING",
        "SQL ": "SQL",
        "STRUTS": "STRUTS",
        "SUBLIM": "SUBLIME TEXT",
        "TABLE": "TABLEAU",
        "TENSORF": "TENSORFLOW",
        "TERADA": "TERADATA",
        "TEST ": "TEST",
        "TIBCO ": "TIBCO",
        "UBUNT": "UBUNTU",
        "UI ": "UI",
        "USER I": "UI",
        "UNIT T": "TEST",
        "UNIX": "UNIX",
        "VISUAL BASIC": "VISUAL BASIC",
        "VMWARE": "VMWARE",
        "VULNERABILITY": "VULNERABILITY MANAGEMENT",
        "WAN ": "WAN",
        "WEB API": "API",
        "WEB ": "WEB",
        "WEBSPHERE": "WEBSPHERE",
        "WINDOWS AZ": "AZURE",
        "WIN ": "WINDOWS",
        "WINDOW": "WINDOWS",
        "ZEND": "ZEND",
        "PERSONAL HOME PAGE": "PHP",
    }

    # Explode each cell separate locations and standardize them
    s = s.str.split(";").explode()
    for k, v in replacement_dict.items():
        s = s.apply(lambda e: replace_if_startswith(e, k, v))

    s = squeeze_series(s)

    return s


def get_eur_amount(amount, currency_code):
    """
    Get EUR equivalent amount based on amount and currency_code.

    Args:
        amount (float): money amount.
        currency_code (str): currency code.

    Returns:
        Equivalent amount in EUR.

    Authors:
        Tue Nguyen
    """
    exchange_rates = {"EUR": 1, "GBP": 1.13, "CHF": 1.02, "USD": 0.91}
    rate = exchange_rates.get(currency_code, 1)

    return rate * amount


def get_eur_amount_series(df, amount_col, currency_col):
    """
    Generates a series with EUR equivalent amount from amount_col
    and currency_col of data frame df

    Args:
        df (data frame): a data frame.
        amount_col (float): name of original amount column.
        currency_col (str): name of currency column
    """
    return df.apply(
        lambda row: get_eur_amount(row[amount_col], row[currency_col]), axis=1
    )


# ==================================================
# CLEAN ORIGINAL DATA
# This section contains master functions for cleaning the original data
# ==================================================


def clean_organizations():
    """
    Cleans organizations (accounts) table.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    src = "data/original/Account_Anonymized.csv"
    dest = "data/cleaned/organizations.feather"
    col_mapping = {
        # Org ID
        "Id": "org_id",
        # Org general info
        "Name": "org_name",
        "Industry": "org_industry",
        "Type": "org_type",
        # Locations
        "Country__c": "org_country",
        "City__c": "org_city",
    }

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = read_raw_data(src, col_mapping, low_memory=False)

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    # Clean locations
    cols = ["org_country", "org_city"]
    for c in cols:
        df[c] = df[c].str.upper()

    # --------------------------------------------------
    # Filter rows
    # --------------------------------------------------
    df = df.dropna(subset=["org_id"]).reset_index(drop=True)

    # Save
    df.to_feather(dest)


def clean_candidates():
    """
    Cleans candidates (contacts) data.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    src = "data/original/Contact_Anonymized.csv"
    dest = "data/cleaned/candidates.feather"
    col_mapping = {
        # General
        "Id": "cand_id",
        # Qualification
        "AVTRRT__Education_Details__c": "cand_edu",
        "AVTRRT__AutoPopulate_Skillset__c": "cand_skills",
        "AVTRRT__IT_Competency__c": "cand_it_skills",
        "AVTRRT__General_Competency__c": "cand_general_skills",
        "AVTRRT__Other_Competency__c": "cand_other_skills",
        # Employment history
        "Title": "cand_current_title",
        "AVTRRT__Current_Employer__c": "cand_current_employer",
        "AVTRRT__Previous_Titles__c": "cand_prev_titles",
        "AVTRRT__Previous_Employers__c": "cand_prev_employers",
        "AVTRRT__Candidate_Summary__c": "cand_summary",
        "AVTRRT__Candidate_Short_List__c": "cand_shortlist",
        # Personal details
        "MailingCountry": "cand_country",
        "MailingCity": "cand_city",
        # Datetime
        "CreatedDate": "cand_created_at",
    }

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = read_raw_data(src, col_mapping, low_memory=False)

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    # Clean skills
    cols = ["cand_skills", "cand_it_skills", "cand_general_skills", "cand_other_skills"]
    for c in cols:
        df[c] = normalize_skills_series_extra(df[c])

    # Current job
    cols = ["cand_current_title", "cand_current_employer"]
    for c in cols:
        df[c] = df[c].str.strip().str.upper()

    # Previous jobs and short list
    cols = ["cand_prev_titles", "cand_prev_employers", "cand_shortlist"]
    for c in cols:
        df[c] = normalize_str_separated_series(df[c])

    # Locations
    df["cand_city"] = normalize_city_series(df["cand_city"])

    # Locations
    df["cand_country"] = normalize_country_series(df["cand_country"])

    # --------------------------------------------------
    # Filter rows
    # --------------------------------------------------
    df = df.dropna(subset=["cand_id"])

    # Save
    df.reset_index(drop=True).to_feather(dest)


def clean_jobs():
    """
    Cleans jobs data.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    src = "data/original/AVTRRT__Job_Anonymized.csv"
    dest = "data/cleaned/jobs.feather"
    col_mapping = {
        # Job ID
        "Id": "job_id",
        # Job general info
        "AVTRRT__Job_Title__c": "job_title",
        "AVTRRT__Job_Description__c": "job_description",
        "AVTRRT__Job_Category__c": "job_category",
        "AVTRRT__Job_Term__c": "job_term",
        "Type_of_contract__c": "job_contract",
        "AVTRRT__Number_of_Positions__c": "job_n_positions",
        "Reason_for_requisition__c": "job_reason",
        # Job requirement
        "AVTRRT__Experience__c": "job_experience",
        "AVTRRT__AutoPopulated_Skills__c": "job_skills",
        "Team_Leading__c": "job_team_lead_size",
        "Project_Management__c": "job_project_management",
        "Travel__c": "job_pct_travel",
        "Languages__c": "job_lang_1",
        "Language_02__c": "job_lang_2",
        "Level_Language_01__c": "job_lang_1_level",
        "Level_Language_02__c": "job_lang_2_level",
        # Compensation
        "CurrencyIsoCode": "job_currency_code",
        "Fix_Salary__c": "job_salary",
        "Variable_Bonus__c": "job_bonus",
        "Benefits__c": "job_benefit",
        # Location
        "Location__c": "job_locations",
        # OSN stuff
        "OSN_Practice__c": "job_osn_practice",
        "Commission_Fee__c": "job_fee",
        # Date time
        "CreatedDate": "job_created_at",
    }

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = read_raw_data(src, col_mapping)

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    # Clean location
    df["job_locations"] = normalize_city_series(df["job_locations"])

    # Clean job skills
    df["job_skills"] = normalize_skills_series(df["job_skills"])

    # Clean money
    df["job_salary"] = get_eur_amount_series(df, "job_salary", "job_currency_code")
    df["job_bonus"] = get_eur_amount_series(
        df, "job_bonus", "job_currency_code"
    ).fillna(0)
    df["job_total_salary"] = df["job_salary"] + df["job_bonus"]
    df.drop(columns=["job_currency_code"], inplace=True)

    # Clean date
    df["job_created_at"] = pd.to_datetime(df["job_created_at"])

    # Clean OSN practice
    df["job_osn_practice"] = df["job_osn_practice"].str.upper()
    other_practices = [np.nan, None, "OSN", "OST"]
    cond = df["job_osn_practice"].isin(other_practices)
    df.loc[cond, "job_osn_practice"] = "OTHER"

    # Clean title
    df["job_title"] = df["job_title"].str.strip().str.upper()

    # Job category
    df["job_category"] = normalize_str_separated_series(df["job_category"], sep="[/,;]")

    # Job benefit
    df["job_benefit"] = normalize_str_separated_series(df["job_benefit"], sep="[/,;]")

    # --------------------------------------------------
    # Filter rows
    # --------------------------------------------------
    df = df.dropna(subset=["job_id"]).reset_index(drop=True)

    # Save
    df.to_feather(dest)


def clean_applications():
    """
    Cleans job applications table.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    src = "data/original/AVTRRT__Job_Applicant__Anonymized.csv"
    dest = "data/cleaned/applications.feather"
    col_mapping = {
        # Application ID
        "Id": "app_id",
        # Account (org) ID -> pointing orgs table
        "AVTRRT__Account_Job__c": "org_id",
        # contact (candidate) ID -> pointing to candidates table
        "AVTRRT__Contact_Candidate__c": "cand_id",
        # Job ID -> pointing to jobs table
        "AVTRRT__Job__c": "job_id",
        # Job related info
        "AVTRRT__Job_Title__c": "app_job_title",
        # Candidate related
        "AVTRRT__Resume__c": "app_cv",
        "Current_Role__c": "app_current_role",
        "Type_of_Contract__c": "app_contract",
        # Location
        "AVTRRT__City__c": "app_city",
        # Offer
        "AVTRRT__Reject_Qualifiers__c": "app_reject_qualifier",
        # Job match processing
        "AVTRRT__Stage__c": "app_stage",
        # Datetime
        "CreatedDate": "app_created_at",
        "AVTRRT__Resume_Received_Date__c": "app_cv_received_at",
    }

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = read_raw_data(src, col_mapping, low_memory=False)

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    # Normalize string cols
    cols = ["app_job_title", "app_current_role", "app_reject_qualifier"]
    for c in cols:
        df[c] = df[c].str.strip().str.upper()

    # City
    df["app_city"] = normalize_city_series(df["app_city"])

    # Date
    cols = ["app_created_at", "app_cv_received_at"]
    for c in cols:
        df[c] = pd.to_datetime(df[c])

    # --------------------------------------------------
    # Filter rows
    # --------------------------------------------------
    df = df.dropna(subset=["app_id"]).reset_index(drop=True)

    # Save
    df.to_feather(dest)


def clean_emp_histories():
    """
    Cleans employment histories.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    src = "data/original/AVTRRT__Employment_History__Anonymized.csv"
    dest = "data/cleaned/histories.feather"
    col_mapping = {
        # Hist ID
        "Id": "hist_id",
        # Candidate ID
        "AVTRRT__Candidate__c": "cand_id",
        # Hist general info
        "AVTRRT__Description__c": "hist_description",
        "AVTRRT__EmployerOrgName__c": "hist_org",
        # Date
        "AVTRRT__Start_Date__c": "hist_start_at",
        "AVTRRT__End_Date__c": "hist_end_at",
        "CreatedDate": "hist_created_at",
    }

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = read_raw_data(src, col_mapping)

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    # Clean dates
    cols = ["hist_start_at", "hist_end_at"]
    for c in cols:
        df[c] = pd.to_datetime(df[c].apply(normalize_date), errors="coerce")

    df["hist_created_at"] = pd.to_datetime(df["hist_created_at"])

    # Add duration in years
    start = df["hist_start_at"].copy()
    end = df["hist_end_at"].copy()
    cond = end.isnull()
    end.loc[cond] = df.loc[cond, "hist_created_at"]
    df["hist_years"] = ((end - start).dt.days / 365.0).round(1)

    # If hist_months == 0 (same start year and end year), fix it to half of a year
    cond = df["hist_years"] == 0
    df.loc[cond, "hist_years"] = 0.5

    # Clean orgs
    df["hist_org"] = df["hist_org"].str.strip().str.upper()

    # --------------------------------------------------
    # Filter rows
    # --------------------------------------------------
    df = df.dropna(subset=["hist_id"]).reset_index(drop=True)

    # Save
    df.to_feather(dest)


def clean_all():
    """
    Cleans all data.

    Returns:
        None.

    Authors:
        Tue Nguyen
    """
    clean_organizations()
    clean_candidates()
    clean_jobs()
    clean_emp_histories()
    clean_applications()
    clean_applications()
