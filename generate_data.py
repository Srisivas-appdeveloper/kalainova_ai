import requests, json, time, os

API_KEY = os.environ["XAI_API_KEY"]

questions = [
    # Widgets
    "What is StatefulWidget in Flutter?",
    "What is StatelessWidget in Flutter?",
    "What is the difference between StatelessWidget and StatefulWidget?",
    "How do I use setState in Flutter?",
    "What is BuildContext in Flutter?",
    "What is the widget lifecycle in Flutter?",
    "How do I create a custom widget in Flutter?",
    "What is InheritedWidget in Flutter?",
    "What is the difference between Container and SizedBox?",
    "What is the difference between Row and Column in Flutter?",
    "What is the difference between Expanded and Flexible in Flutter?",
    "How do I use Stack widget in Flutter?",
    "How do I use ListView in Flutter?",
    "How do I use GridView in Flutter?",
    "What is MediaQuery in Flutter?",
    "How do I use GestureDetector in Flutter?",
    "What is the difference between GestureDetector and InkWell?",
    "How do I create a bottom navigation bar in Flutter?",
    "How do I create an AppBar in Flutter?",
    "How do I use Scaffold in Flutter?",
    "How do I use TabBar in Flutter?",
    "How do I use Drawer in Flutter?",
    "How do I use Dialog in Flutter?",
    "How do I use SnackBar in Flutter?",
    "How do I use TextField in Flutter?",
    "How do I use Form and FormField in Flutter?",
    "How do I validate a form in Flutter?",
    "How do I use DropdownButton in Flutter?",
    "How do I use Checkbox in Flutter?",
    "How do I use Switch in Flutter?",
    "How do I use Slider in Flutter?",
    "How do I use Image widget in Flutter?",
    "How do I use CircleAvatar in Flutter?",
    "How do I use Card widget in Flutter?",
    "How do I use ListTile in Flutter?",
    "How do I use Padding widget in Flutter?",
    "How do I use Align widget in Flutter?",
    "How do I use Center widget in Flutter?",
    "How do I use Wrap widget in Flutter?",
    "How do I use SingleChildScrollView in Flutter?",
    # Navigation
    "How do I navigate between screens in Flutter?",
    "How do I use go_router in Flutter?",
    "How do I pass data between screens in Flutter?",
    "How do I implement deep linking in Flutter?",
    "How do I use named routes in Flutter?",
    "How do I pop back to previous screen in Flutter?",
    "How do I replace current screen in Flutter?",
    # State Management
    "What is Provider in Flutter?",
    "What is GetX in Flutter?",
    "What is Riverpod in Flutter?",
    "What is BLoC pattern in Flutter?",
    "What is the difference between Provider and GetX?",
    "How do I use ChangeNotifier in Flutter?",
    "How do I use ValueNotifier in Flutter?",
    "How do I use StreamController in Flutter?",
    "What is Cubit in Flutter BLoC?",
    "How do I use context.watch in Flutter?",
    # Async
    "What is Dart async/await?",
    "What is a Future in Dart?",
    "What is a Stream in Dart?",
    "How do I use FutureBuilder in Flutter?",
    "How do I use StreamBuilder in Flutter?",
    "How do I handle errors in Future in Dart?",
    "How do I use isolates in Dart?",
    "What is compute() in Flutter?",
    # Networking
    "How do I make HTTP requests in Flutter?",
    "How do I use dio package in Flutter?",
    "How do I parse JSON in Flutter?",
    "How do I handle API errors in Flutter?",
    "How do I add headers to HTTP requests in Flutter?",
    "How do I implement authentication tokens in Flutter?",
    "How do I use interceptors in dio?",
    # Storage
    "How do I use SharedPreferences in Flutter?",
    "How do I use Hive database in Flutter?",
    "How do I use SQLite in Flutter?",
    "How do I store files in Flutter?",
    "How do I use secure storage in Flutter?",
    # Firebase
    "How do I use Firebase in Flutter?",
    "How do I implement Firebase Auth in Flutter?",
    "How do I use Firestore in Flutter?",
    "How do I use Firebase Storage in Flutter?",
    "How do I use Firebase Cloud Messaging in Flutter?",
    # Animation
    "How do I add animations in Flutter?",
    "What is AnimationController in Flutter?",
    "How do I use Hero animation in Flutter?",
    "How do I use Tween in Flutter?",
    "What is AnimatedBuilder in Flutter?",
    "How do I use implicit animations in Flutter?",
    # Dart
    "What is the difference between final and const in Dart?",
    "What is null safety in Dart?",
    "How do I use mixins in Dart?",
    "What is the difference between abstract class and interface in Dart?",
    "How do I use extension methods in Dart?",
    "What is the difference between Map and List in Dart?",
    "How do I use generics in Dart?",
    "What is a factory constructor in Dart?",
    "How do I use named constructors in Dart?",
    "What is the difference between == and identical() in Dart?",
    "How do I use spread operator in Dart?",
    "What is late keyword in Dart?",
    "How do I use typedef in Dart?",
    "What is cascade notation in Dart?",
    "How do I sort a list in Dart?",
    "How do I filter a list in Dart?",
    "How do I map a list in Dart?",
    "How do I use Set in Dart?",
    "How do I use Map in Dart?",
    "What is the difference between dynamic and Object in Dart?",
    # Architecture
    "What is Clean Architecture in Flutter?",
    "What is MVVM in Flutter?",
    "How do I implement repository pattern in Flutter?",
    "How do I use dependency injection in Flutter?",
    "What is GetIt in Flutter?",
    "How do I use injectable in Flutter?",
    # Testing
    "How do I write unit tests in Flutter?",
    "How do I write widget tests in Flutter?",
    "How do I write integration tests in Flutter?",
    "How do I mock dependencies in Flutter tests?",
    "How do I use Mockito in Flutter?",
    # Performance
    "How do I optimize Flutter app performance?",
    "What is const constructor in Flutter?",
    "How do I use RepaintBoundary in Flutter?",
    "How do I reduce rebuilds in Flutter?",
    "What is the difference between keys in Flutter?",
    # Platform
    "How do I implement push notifications in Flutter?",
    "How do I use platform channels in Flutter?",
    "How do I access camera in Flutter?",
    "How do I access location in Flutter?",
    "How do I implement biometric authentication in Flutter?",
    "How do I use local notifications in Flutter?",
    "How do I open a URL in Flutter?",
    "How do I share content in Flutter?",
    # Build & Deploy
    "What is Flutter flavors?",
    "How do I build a release APK in Flutter?",
    "How do I sign an Android app in Flutter?",
    "How do I configure app permissions in Flutter?",
    "How do I use environment variables in Flutter?",
    "What is the difference between debug and release mode in Flutter?",
    # Misc
    "What is hot reload in Flutter?",
    "What is the difference between hot reload and hot restart?",
    "How do I implement dark mode in Flutter?",
    "How do I make a Flutter app responsive?",
    "How do I use MediaQuery for responsive design?",
    "How do I create a Flutter package?",
    "How do I use pubspec.yaml in Flutter?",
    "What is the difference between runApp and main in Flutter?",
    "How do I use ThemeData in Flutter?",
    "How do I customize fonts in Flutter?",
    "How do I add assets to Flutter app?",
    "How do I use localization in Flutter?",
    "What is the difference between Stateful and Stateless lifecycle?",
    "How do I use LayoutBuilder in Flutter?",
    "How do I use CustomPainter in Flutter?",
    "What is RenderObject in Flutter?",
    "How do I implement search functionality in Flutter?",
    "How do I implement pagination in Flutter?",
    "How do I use RefreshIndicator in Flutter?",
    "How do I implement infinite scroll in Flutter?",
]

results = []
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q[:50]}...")
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json={
                "model": "grok-3-mini",
                "messages": [
                    {"role": "system", "content": "You are a senior Flutter developer. Give accurate, clear answers with working code examples."},
                    {"role": "user", "content": q}
                ],
                "max_tokens": 600
            }
        )
        answer = resp.json()['choices'][0]['message']['content']
        results.append({"q": q, "a": answer})
        print(f"  OK: {len(answer)} chars")
        time.sleep(0.3)
    except Exception as e:
        print(f"  Error: {e}")
        time.sleep(1)

with open('/Users/technotackle/distributed_ai/models/grok_flutter_qa.txt', 'w') as f:
    for item in results:
        f.write(f"### Question\n{item['q']}\n\n### Answer\n{item['a']}\n\n")

print(f"\nDone! Saved {len(results)} Q&A pairs")
