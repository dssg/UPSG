from upsg import Pipeline
from upsg.import import CSVRead
from upsg.transform import Impute
from upsg.export import CSVWrite

p = Pipeline()

csv_read_uid = p.add(CSVRead('sample_in.csv'))
impute_uid = p.add(Impute(lambda column: 0))
csv_write_uid = p.add(CSVWrite('sample_out.csv'))

p.connect(csv_read_uid, 'out', impute_uid, 'in')
p.connect(imput_uid, 'out', csv_write_uid)

p.run()
