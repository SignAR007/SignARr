(() => {
  'use strict';
  // Vocabulary supplied for the project. These are proposed word/translation pairs,
  // not verified sign forms; don't infer gesture data from the glosses.
  const groups = [
    ['Greetings and basic communication', `Hello|Kumusta
Good morning|Magandang umaga
Good afternoon|Magandang hapon
Good evening|Magandang gabi
Good night|Magandang gabi
Goodbye|Paalam
Thank you|Salamat
You're welcome|Walang anuman
Please|Pakiusap
Sorry|Pasensya
How are you?|Kumusta ka?
I am fine|Mabuti ako
Nice to meet you|Ikinagagalak kitang makilala
See you tomorrow|Kita tayo bukas
Welcome|Maligayang pagdating`],
    ['Common questions and responses', `What?|Ano?
Who?|Sino?
Where?|Saan?
When?|Kailan?
Why?|Bakit?
How?|Paano?
How much?|Magkano?
How many?|Ilan?
What is your name?|Ano ang pangalan mo?
Do you understand?|Naiintindihan mo ba?
Do you know?|Alam mo ba?
Can you help me?|Maaari mo ba akong tulungan?
Are you okay?|Ayos ka lang ba?
What happened?|Anong nangyari?
Where are you going?|Saan ka pupunta?`],
    ['Essential daily expressions', `Yes|Oo
No|Hindi
Maybe|Siguro
Okay|Sige
I understand|Naiintindihan ko
I don't understand|Hindi ko naiintindihan
I know|Alam ko
I don't know|Hindi ko alam
Wait|Sandali
Stop|Tigil
Come here|Halika rito
Go|Pumunta
Let's go|Tara na
Repeat|Ulitin
I need help|Kailangan ko ng tulong`],
    ['Family members and relationships', `Family|Pamilya
Father|Tatay
Mother|Nanay
Brother|Kuya / Kapatid na lalaki
Sister|Ate / Kapatid na babae
Grandfather|Lolo
Grandmother|Lola
Son|Anak na lalaki
Daughter|Anak na babae
Baby|Sanggol
Friend|Kaibigan
Husband|Asawa
Wife|Asawa
Child|Bata
Teacher|Guro`],
    ['Food and drinks', `Eat|Kain
Drink|Inom
Water|Tubig
Rice|Kanin
Bread|Tinapay
Milk|Gatas
Coffee|Kape
Juice|Katas
Food|Pagkain
Hungry|Gutom
Thirsty|Uhaw
Breakfast|Almusal
Lunch|Tanghalian
Dinner|Hapunan
Delicious|Masarap`],
    ['Places and directions', `Home|Bahay
School|Paaralan
Hospital|Ospital
Bathroom|Banyo
Classroom|Silid-aralan
Office|Opisina
Market|Palengke
Store|Tindahan
Church|Simbahan
Left|Kaliwa
Right|Kanan
Straight|Diretso
Near|Malapit
Far|Malayo
Here|Dito`],
    ['Health and body parts', `Sick|May sakit
Pain|Sakit
Headache|Sakit ng ulo
Fever|Lagnat
Cough|Ubo
Medicine|Gamot
Doctor|Doktor
Nurse|Nars
Hospital|Ospital
Help me|Tulungan ako
Head|Ulo
Eyes|Mata
Ears|Tainga
Mouth|Bibig
Hand|Kamay`],
    ['School and education', `Student|Estudyante
Teacher|Guro
School|Paaralan
Book|Aklat
Pen|Panulat
Paper|Papel
Learn|Matuto
Study|Mag-aral
Read|Magbasa
Write|Magsulat
Question|Tanong
Answer|Sagot
Assignment|Takdang-aralin
Exam|Pagsusulit
Understand|Intindihin`],
    ['Transportation and travel', `Car|Kotse
Bus|Bus
Jeepney|Dyip
Motorcycle|Motorsiklo
Bicycle|Bisikleta
Train|Tren
Airplane|Eroplano
Boat|Bangka
Ride|Sakay
Drive|Magmaneho
Travel|Biyahe
Stop|Hinto
Ticket|Tiket
Fare|Pamasahe
Where is the station?|Nasaan ang istasyon?`],
    ['Emotions and feelings', `Happy|Masaya
Sad|Malungkot
Angry|Galit
Afraid|Takot
Tired|Pagod
Excited|Nasasabik
Worried|Nag-aalala
Confused|Nalilito
Love|Pagmamahal
Like|Gusto
Dislike|Ayaw
Bored|Nababagot
Lonely|Malungkot / Nag-iisa
Surprised|Nagulat
Fine|Ayos`],
    ['Numbers and time', `One|Isa
Two|Dalawa
Three|Tatlo
Four|Apat
Five|Lima
Six|Anim
Seven|Pito
Eight|Walo
Nine|Siyam
Ten|Sampu
Today|Ngayon
Tomorrow|Bukas
Yesterday|Kahapon
Morning|Umaga
Afternoon|Hapon
Night|Gabi
Now|Ngayon
Later|Mamaya
Early|Maaga
Late|Huli`],
    ['Emergency and safety', `Emergency|Emerhensiya
Danger|Panganib
Fire|Sunog
Accident|Aksidente
Police|Pulis
Ambulance|Ambulansya
Help|Tulong
Call|Tumawag
Be careful|Mag-ingat
Lost|Nawawala
Safe|Ligtas
Stop!|Tigil!
Don't touch|Huwag hawakan
I need a doctor|Kailangan ko ng doktor
Call the police|Tumawag ng pulis`]
  ];
  const entries = [];
  groups.forEach(([category, lines]) => lines.split('\n').forEach(line => {
    const [english, filipino] = line.split('|');
    entries.push({ id: `FSL${String(entries.length + 1).padStart(3, '0')}`, english, filipino, category });
  }));

  const list = document.getElementById('signList');
  const search = document.getElementById('search');
  const category = document.getElementById('category');
  const count = document.getElementById('resultCount');
  const empty = document.getElementById('emptyState');
  for (const [name] of groups) {
    const option = document.createElement('option'); option.value = name; option.textContent = name; category.append(option);
  }
  function render() {
    const query = search.value.trim().toLocaleLowerCase();
    const selected = category.value;
    const matches = entries.filter(entry => (selected === 'all' || entry.category === selected)
      && (!query || `${entry.english} ${entry.filipino} ${entry.id}`.toLocaleLowerCase().includes(query)));
    count.textContent = `${matches.length} of ${entries.length} proposed entries`;
    empty.hidden = matches.length > 0;
    const fragment = document.createDocumentFragment();
    for (const entry of matches) {
      const details = document.createElement('details'); details.className = 'sign-card';
      const summary = document.createElement('summary'); summary.className = 'sign-summary';
      const words = document.createElement('span');
      const title = document.createElement('strong'); title.className = 'sign-title'; title.textContent = entry.english;
      const translation = document.createElement('span'); translation.className = 'translation'; translation.textContent = entry.filipino;
      const id = document.createElement('small'); id.className = 'sign-id'; id.textContent = `${entry.id} · ${entry.category}`;
      words.append(title, translation, id);
      const status = document.createElement('span'); status.className = 'status'; status.textContent = 'Pending verification';
      summary.append(words, status);
      const body = document.createElement('div'); body.className = 'details-body';
      const model = document.createElement('div'); model.className = 'model-placeholder';
      model.innerHTML = '<div><strong>2D sign model not available yet</strong>No reviewed gesture reference or landmark record has been supplied for this entry.</div>';
      const note = document.createElement('p'); note.textContent = 'Recognition status: not trained. This English–Filipino vocabulary entry does not establish the correct FSL handshape, movement, facial expression, or grammar.';
      body.append(model, note); details.append(summary, body); fragment.append(details);
    }
    list.replaceChildren(fragment);
  }
  search.addEventListener('input', render);
  category.addEventListener('change', render);
  render();
})();
