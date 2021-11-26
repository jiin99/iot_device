# iot_device
Fire detection project

### Purpose
- early detection
- no false alarm

### Enter DB

```bash
mysql -u $id -p --port $port --host $ip
```
### Download data directly
```bash
mysql -u $id -p --port $port --host $ip  -D $instance -e "select * from $table_name" | tr '\t' ',' > 'my_table.csv'
```
### Data sample

<img width="500" alt="image" src="https://user-images.githubusercontent.com/62350977/143509188-7f729a0f-ee48-4467-bc38-cbbbd5afbe7a.png">
