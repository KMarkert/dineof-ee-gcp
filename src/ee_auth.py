import ee
import google.auth

def get_ee_credentials():
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )

    return credentials


def ee_initialize(project):
    credentials = get_ee_credentials()
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url='https://earthengine-highvolume.googleapis.com',
    )
    
    return