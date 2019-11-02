## API 

### Get recommended books

- *Method* : POST

- *Request*:

```json

    {"book_id": 1,
      "book_title": "title",
      "abstract": "Lorem ipsm bla bla",
      "json_file": "/path/to/json"
    }
```

- *Response*:

```json
   [
    {
    "book_id": 2,
    "book_title": "Title"
    },
    {
    "book_id": 2,
    "book_title": "Title"
    },
    {
    "book_id": 2,
    "book_title": "Title"
    },
    {
    "book_id": 2,
    "book_title": "Title"
    },
    {
    "book_id": 2,
    "book_title": "Title"
    }
   ] 
```

---
