import struct
import matplotlib.pyplot as plt
from pathlib import Path
from mcap.reader import make_reader
from rosbags.typesys import stores, get_typestore

def print_topics(mcap_file: Path) -> None:
    """
    Přečte metadata na konci souboru a vypíše všechny topicy a počty zpráv.
    """
    print(f"--- Analyzuji soubor: {mcap_file.name} ---")
    with open(mcap_file, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        
        if summary and summary.statistics:
            print("Nalezené topicy a počty zpráv:")
            print("-" * 50)
            for channel_id, count in summary.statistics.channel_message_counts.items():
                channel = summary.channels[channel_id]
                schema = summary.schemas[channel.schema_id]
                
                print(f"Topic: {channel.topic}")
                print(f"  - Typ zprávy: {schema.name}")
                print(f"  - Počet zpráv: {count}")
                print("-" * 50)
        else:
            print("Soubor neobsahuje metadata (summary).")

def print_topic_data(mcap_file: Path, target_topic: str) -> None:
    """
    Přečte zprávy, odřízne obálku loggeru a vypíše data do terminálu.
    """
    print(f"\n--- Čtu data z topicu: {target_topic} ---")
    typestore = get_typestore(stores.Stores.ROS2_HUMBLE)
    
    with open(mcap_file, 'rb') as f:
        reader = make_reader(f)
        msg_count = 0
        
        for schema, channel, message in reader.iter_messages(topics=[target_topic]):
            try:
                str_len = struct.unpack('<H', message.data[:2])[0]
                payload_offset = 2 + str_len + 4
                real_cdr_data = message.data[payload_offset:]
                
                ros_msg = typestore.deserialize_cdr(real_cdr_data, schema.name)
                timestamp_sec = message.log_time / 1e9
                
                print(f"Čas: {timestamp_sec:.3f} s | Obsah:\n{ros_msg}")
                
                msg_count += 1
                if msg_count >= 10:
                    print("... (výpis zastaven po 10 zprávách) ...")
                    break
                    
            except Exception as e:
                print(f"Chyba při dekódování zprávy: {e}")
                break

def plot_topic_data(mcap_file: Path, target_topic: str) -> None:
    """
    Vyfiltruje data z daného topicu, dynamicky vybere sledované hodnoty 
    (podle typu zprávy) a vykreslí je v závislosti na čase.
    """
    print(f"\n--- Vykresluji data z topicu: {target_topic} ---")
    typestore = get_typestore(stores.Stores.ROS2_HUMBLE)
    
    timestamps = []
    # Slovník pro hodnoty (klíč = název v legendě, hodnota = seznam dat)
    plot_data = {} 
    
    with open(mcap_file, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages(topics=[target_topic]):
            try:
                # 1. Odstranění nestandardní obálky
                str_len = struct.unpack('<H', message.data[:2])[0]
                payload_offset = 2 + str_len + 4
                real_cdr_data = message.data[payload_offset:]
                
                # 2. Dekódování čistých dat
                ros_msg = typestore.deserialize_cdr(real_cdr_data, schema.name)
                
                # Uložení času v sekundách
                timestamp_sec = message.log_time / 1e9
                timestamps.append(timestamp_sec)
                
                # 3. Rozpoznání datového typu a uložení patřičných hodnot
                if 'Pose' in schema.name:
                    # Podporuje Pose i PoseStamped
                    pos = ros_msg.pose.position if hasattr(ros_msg, 'pose') else ros_msg.position
                    
                    if 'Poloha X' not in plot_data: plot_data['Poloha X'] = []
                    if 'Poloha Y' not in plot_data: plot_data['Poloha Y'] = []
                    plot_data['Poloha X'].append(pos.x)
                    plot_data['Poloha Y'].append(pos.y)
                    
                elif 'Twist' in schema.name:
                    # Podporuje Twist i TwistStamped
                    tw = ros_msg.twist if hasattr(ros_msg, 'twist') else ros_msg
                    
                    if 'Lineární rychlost X' not in plot_data: plot_data['Lineární rychlost X'] = []
                    if 'Úhlová rychlost Z' not in plot_data: plot_data['Úhlová rychlost Z'] = []
                    plot_data['Lineární rychlost X'].append(tw.linear.x)
                    plot_data['Úhlová rychlost Z'].append(tw.angular.z)
                    
                elif 'Float64' in schema.name or 'Int16' in schema.name:
                    # Jednoduché číselné zprávy (např. od kol)
                    if 'Hodnota' not in plot_data: plot_data['Hodnota'] = []
                    plot_data['Hodnota'].append(ros_msg.data)
                    
                else:
                    print(f"Typ zprávy '{schema.name}' není zatím pro graf podporován.")
                    return
                
            except Exception as e:
                print(f"Chyba při dekódování zprávy: {e}")
                return
                
    if not timestamps:
        print(f"V topicu {target_topic} nebyla nalezena žádná data.")
        return

    # Normalizace času (aby graf začínal od nuly)
    start_time = timestamps[0]
    normalized_timestamps = [t - start_time for t in timestamps]
    
    # 4. Samotné vykreslení
    plt.figure(figsize=(10, 6))
    for label, values in plot_data.items():
        plt.plot(normalized_timestamps, values, label=label)
        
    plt.title(f"Průběh dat v čase: {target_topic}")
    plt.xlabel("Čas od začátku záznamu (s)")
    plt.ylabel("Hodnota")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cesta_k_souboru = Path('../log_files/session_20260908_201041.mcap')
    
    # 1. Výpis všech topiců
    print_topics(cesta_k_souboru)
    
    # 2. Výpis dat do konzole (např. gps/pose)
    # print_topic_data(cesta_k_souboru, '/imu')
    
    # 3. Vykreslení grafu – schválně zkus topicy různých typů!
    # plot_topic_data(cesta_k_souboru, '/gps/pose')
    # plot_topic_data(cesta_k_souboru, '/map/pose')

    # plot_topic_data(cesta_k_souboru, '/servo/front/angle_req')