import boto3
import pandas

batch_results = pandas.read_csv('Batch_5093535_batch_results.csv')#.set_index(['HITId', 'WorkerId'])

MTURK_ACCESS_KEY = os.environ["MTURK_ACCESS_KEY"]
MTURK_SECRET = os.environ["MTURK_SECRET"]
MTURK_SANDBOX_URL = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

def mturk_client(live_hit=True):
    if live_hit:
        mturk = boto3.client('mturk',
           aws_access_key_id = MTURK_ACCESS_KEY,
           aws_secret_access_key = MTURK_SECRET,
           region_name='us-east-1'
           )
    else:
        mturk = boto3.client('mturk',
          aws_access_key_id = MTURK_ACCESS_KEY,
          aws_secret_access_key = MTURK_SECRET,
          region_name='us-east-1',
          endpoint_url=MTURK_SANDBOX_URL
        )
    return mturk

mturk = mturk_client(True)

def bonus(live_hit, worker_id, assignment_id, bonus_amount, reason):
    mturk = mturk_client(live_hit)
    response = mturk.send_bonus(WorkerId=worker_id, AssignmentId=assignment_id, BonusAmount=bonus_amount, Reason=reason)
    if ("ResponseMetadata" not in response or 
            "HTTPStatusCode" not in response["ResponseMetadata"] or
            response["ResponseMetadata"]["HTTPStatusCode"] != 200
           ):
        print("ERROR: Bonus payment to worker {} failed!", worker_id)
        print(response)
        print()


for row in batch_results[["WorkerId", "AssignmentId"]].iloc:
    worker_id = row[0]
    assignment_id = row[1]
    bonus(True, worker_id, assignment_id, "0.15", "visual dialogue game")
    import pdb; pdb.set_trace()
