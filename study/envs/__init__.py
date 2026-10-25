from study.envs.construction import ConstructionEnvironment
from study.envs.hospital_network import HospitalNetworkEnvironment
from study.envs.jira_multiday import JiraMultiDayEnvironment

ENVIRONMENTS = {
    "jira": JiraMultiDayEnvironment,
    "construction": ConstructionEnvironment,
    "hospital": HospitalNetworkEnvironment,
}
